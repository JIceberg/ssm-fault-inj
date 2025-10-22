import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

np.random.seed(42)
torch.random.manual_seed(42)

class Correction_Module_dense(nn.Module):
    def __init__(self):
        super(Correction_Module_dense, self).__init__()
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}
        self.k = 4

    def get_gradient(self, x):
        convolution_nn = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=3,padding='same',padding_mode='circular')
        convolution_nn.weight.requires_grad = False 
        convolution_nn.bias.requires_grad = False 
        convolution_nn.weight[0] = torch.tensor([-1,1,0],dtype=torch.float)
        convolution_nn.bias[0] = torch.tensor([0],dtype=torch.float)
        convolution_nn = convolution_nn.to(x.device)

        grad = []
        for batchind in range(0, x.shape[0]):
            gradind = convolution_nn(x[batchind,:].reshape(1,1,x.shape[1])).reshape(1,x.shape[1])
            grad.append(gradind)

        return torch.stack(grad, dim=0).view(-1, x.shape[1])
    
    def compute_grad(self, x, layer_name):
        if layer_name not in self.num_updates:
            self.num_updates[layer_name] = 0
        self.num_updates[layer_name] += 1

        x = nan_checker(x)
        grad_Y = self.get_gradient(x)
        mean = grad_Y.mean(dim=0).cpu().detach().numpy()
        var = grad_Y.var(dim=0).cpu().detach().numpy()

        if layer_name not in self.mean_grad:
            self.mean_grad[layer_name] = mean
            self.var_grad[layer_name] = var
        else:
            self.mean_grad[layer_name] += (mean - self.mean_grad[layer_name]) / self.num_updates[layer_name]
            self.var_grad[layer_name] += (var - self.var_grad[layer_name]) / self.num_updates[layer_name]

    def forward(self, output, layer_name):
        if self.mean_grad[layer_name] is None or self.var_grad[layer_name] is None:
            return output

        batch_size, num_neurons = output.shape
        output = nan_checker(output)
        
        std_grad = np.sqrt(self.var_grad[layer_name])

        mean_grad_tensor = torch.tensor(self.mean_grad[layer_name], device=output.device)
        std_grad_tensor = torch.tensor(std_grad, device=output.device)

        lower_bound = mean_grad_tensor - self.k * std_grad_tensor
        upper_bound = mean_grad_tensor + self.k * std_grad_tensor

        grad_Y = self.get_gradient(output)
        mask = (grad_Y < lower_bound) | (grad_Y > upper_bound)

        new_output = output.clone()
        new_output[mask] = 0

        # ground_truth = layer_func(input)
        # print("old layer:", output[0, mask[0]])
        # print("new layer:", new_output[0, mask[0]])
        # print("true layer:", ground_truth[0, mask[0]])

        return new_output
    
class Correction_Module_conv(nn.Module):
    def __init__(self):
        super(Correction_Module_conv, self).__init__()
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}
        self.k = 6

    def get_gradient(self, x):
        convolution_nn = nn.Conv1d(in_channels=x.shape[2],out_channels=x.shape[2],kernel_size=3,
            padding='same',padding_mode='circular')
        convolution_nn.weight.requires_grad = False 
        convolution_nn.bias.requires_grad = False 
        convolution_nn.weight[0] = torch.tensor([-1,1,0],dtype=torch.float)
        convolution_nn.bias[0] = torch.tensor([0],dtype=torch.float)
        convolution_nn = convolution_nn.to(x.device)

        grads = []
        for batchind in range(0, x.shape[0]):
            outtemp = torch.swapaxes(x[batchind,:,:,:], 0, 2)
            tempout_test = convolution_nn(outtemp)
            grad = torch.swapaxes(tempout_test, 0, 2)
            grads.append(grad)
        
        grads = torch.stack(grads, dim=0)
        return grads
    
    def compute_grad(self, x, layer_name):
        if layer_name not in self.num_updates:
            self.num_updates[layer_name] = 0
        self.num_updates[layer_name] += 1

        x = nan_checker(x)

        grad_Y = self.get_gradient(x)
        
        mean = grad_Y.mean(dim=0).cpu().detach().numpy()
        var = grad_Y.var(dim=0).cpu().detach().numpy()
        if layer_name not in self.mean_grad:
            self.mean_grad[layer_name] = mean
            self.var_grad[layer_name] = var
        else:
            self.mean_grad[layer_name] += (mean - self.mean_grad[layer_name]) / self.num_updates[layer_name]
            self.var_grad[layer_name] += (var - self.var_grad[layer_name]) / self.num_updates[layer_name]

    def forward(self, output, layer_name):
        if self.mean_grad[layer_name] is None or self.var_grad[layer_name] is None:
            return output

        output = nan_checker(output)
        
        std_grad = np.sqrt(self.var_grad[layer_name])

        mean_grad_tensor = torch.tensor(self.mean_grad[layer_name], device=output.device)
        std_grad_tensor = torch.tensor(std_grad, device=output.device)

        lower_bound = mean_grad_tensor - self.k * std_grad_tensor
        upper_bound = mean_grad_tensor + self.k * std_grad_tensor

        grad_Y = self.get_gradient(output)
        mask = (grad_Y < lower_bound) | (grad_Y > upper_bound)

        new_output = output.clone()
        new_output[mask] = 0

        return new_output

def nan_checker(x):
    nan_check = torch.isnan(x)
    inf_check = torch.isinf(x)
    if torch.sum(nan_check) or torch.sum(inf_check):
        x = x.masked_fill_(nan_check,0)
        x = x.masked_fill_(inf_check,0)
    return x  

def flip_bits(A, error_rate=1e-4, clamp_val=1e6):
    # Flatten tensor for easier manipulation
    flat_output = A.view(-1)

    # Convert float tensor to int representation (IEEE 754)
    float_bits = flat_output.to(torch.float32).cpu().numpy().view(np.uint32)

    # Randomly select bits to flip
    num_elements = flat_output.numel()
    random_bits = np.random.randint(0, 32, size=num_elements, dtype=np.uint32)

    # Create a mask to determine which values to flip
    flip_mask = np.random.rand(num_elements) < error_rate
    does_flip = np.any(flip_mask)
    # print("we flippin?", np.any(flip_mask))

    # Perform bitwise XOR only for selected neurons
    flipped_bits = float_bits ^ (1 << random_bits)

    # Replace only values where flip_mask is True
    float_bits[flip_mask] = flipped_bits[flip_mask]

    # Convert back to PyTorch tensor
    modified_output = torch.tensor(float_bits.view(np.float32), dtype=torch.float32, device=A.device).view(A.shape)
    
    # clamp and disallow nan
    modified_output = torch.nan_to_num(modified_output, nan=0.0, posinf=clamp_val, neginf=-clamp_val)
    
    return modified_output, does_flip

# ---------- SSM Layer ----------
class SSM(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, error_rate=1e-4, inject=False):
        B, L, D = x.shape
        device = x.device
        state = torch.zeros(B, self.d_state, device=device)

        outputs = []
        for t in range(L):
            u_t = x[:, t, :]
            x_t = state @ self.A.T + u_t @ self.B.T

            if not self.training and inject:
                x_t, _ = flip_bits(x_t, error_rate=error_rate)

            state = x_t
            y_t = state @ self.C.T
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        activation = self.activation(y)
        norm = self.norm(activation)
        return torch.nan_to_num(norm, nan=0.0, posinf=1e6, neginf=-1e6)


# ---------- Classifier Model ----------
class SSMClassifier(nn.Module):
    def __init__(self, d_model=128, d_state=64, n_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(28, d_model)
        self.ssm = SSM(d_model=d_model, d_state=d_state)
        self.corr_dense = Correction_Module_dense()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, error_rate=1e-4, compute_grad=False, inject=False):
        x = x.squeeze(1)                # (B, 28, 28)
        x = self.input_proj(x)          # (B, 28, d_model)
        x = self.ssm(x, error_rate=error_rate, inject=inject)
        x = x.transpose(1, 2)           # (B, d_model, 28)
        x = self.pool(x).squeeze(-1)    # (B, d_model)

        pooled = x.clone().detach()

        if not self.training:
            if inject:
                x, _ = flip_bits(x, error_rate=error_rate)

        return self.fc(x), pooled


# ---------- Training Script ----------
def train_model(epochs=5, batch_size=64, lr=1e-3, device="cuda"):
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    # Calculate lengths for 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split the training dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model, optimizer, loss
    model = SSMClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            # stats
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=False):
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        test_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")

    return model, test_loader

def collect_neuron_activations(model, data_loader, device="cuda", inject=False, error_rate=1e-3):
    model.eval()
    pooled_outputs = []
    with torch.no_grad():
        for x, _ in tqdm(data_loader, desc="Collecting clean pooled outputs"):
            x = x.to(device)
            # disable all noise/injection/correction
            _, pooled = model(x, inject=inject, error_rate=error_rate)
            pooled_outputs.append(pooled.cpu().numpy())
    pooled_outputs = np.concatenate(pooled_outputs, axis=0)  # (N, d_model)
    return pooled_outputs

def cluster_neurons_hierarchical(
    activations,          # shape (n_samples, n_neurons)
    metric='correlation',
    linkage_method='average',
    distance_threshold=0.5,
    max_clusters=20
):
    """
    activations: (num_samples, n_neurons)
    Returns: dict with neuron labels, centroids, and per-cluster stats
    """
    X = activations.T.astype(np.float64)  # (n_neurons, n_samples)

    # Normalize per neuron
    neuron_means = X.mean(axis=1, keepdims=True)
    neuron_stds = X.std(axis=1, keepdims=True) + 1e-8
    X_norm = (X - neuron_means) / neuron_stds

    # Hierarchical clustering
    D = pdist(X_norm, metric=metric)
    Z = linkage(D, method=linkage_method)
    labels = fcluster(Z, t=distance_threshold, criterion='distance') - 1

    # Compute per-cluster centroids and dist stats
    centroids, cluster_stats = {}, {}
    for c in np.unique(labels):
        neuron_ids = np.where(labels == c)[0]
        cluster_vectors = X_norm[neuron_ids]
        centroid = cluster_vectors.mean(axis=0)
        dists = np.linalg.norm(cluster_vectors - centroid, axis=1)
        centroids[c] = centroid
        cluster_stats[c] = {
            'mean_dist': np.mean(dists),
            'std_dist': np.std(dists),
            'centroid': centroid,
        }

    clustering_result = {
        'labels': labels,
        'means': neuron_means.flatten(),
        'stds': neuron_stds.flatten(),
    }
    return clustering_result, centroids, cluster_stats

def check_faulty_batch(batch_activations, clustering_result, centroids, cluster_stats, factor=1):
    """
    batch_activations: (B, n_neurons)
    Returns: boolean mask of shape (n_neurons,)
    """
    labels = clustering_result['labels']
    means = clustering_result['means']
    stds = clustering_result['stds']

    # Normalize batch per neuron
    X_norm = (batch_activations - means) / stds  # (B, n_neurons)
    batch_mean_vector = np.nanmean(X_norm, axis=0)

    anomaly_mask = np.zeros(X_norm.shape[1], dtype=bool)

    for c in np.unique(labels):
        neuron_ids = np.where(labels == c)[0]
        if len(neuron_ids) == 0:
            continue

        stats = cluster_stats[c]
        centroid = stats['centroid']
        mean_dist = stats['mean_dist']
        std_dist = stats['std_dist']

        # Distance of current batch’s mean neuron vector to cluster centroid
        dist = np.linalg.norm(batch_mean_vector[neuron_ids] - centroid[neuron_ids])

        # Anomaly if distance exceeds mean + factor*std
        anomaly_mask[neuron_ids] = dist > (mean_dist + factor * std_dist)

    return anomaly_mask

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, test_loader = train_model(epochs=5, device=device)

    clean_activations = collect_neuron_activations(model, test_loader, device=device, inject=False)
    clustering_result, centroids, cluster_stats = cluster_neurons_hierarchical(
        clean_activations,
        metric='correlation',
        linkage_method='average',
        distance_threshold=0.5
    )
    print("Cluster labels for neurons:", clustering_result['labels'])
    print("Means for normalization:", clustering_result['means'].shape, clustering_result['means'])

    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)
    
    model.eval()
    with torch.no_grad():
        logits, batch_activations = model(x_batch, inject=True, error_rate=1e-3)
        batch_activations = batch_activations.cpu().numpy()

    print("Batch activations shape:", batch_activations.shape)
    print("Activations:", batch_activations)

    anomaly_mask = check_faulty_batch(
        batch_activations,
        clustering_result,
        centroids,
        cluster_stats
    )
    print("Anomaly mask for neurons in batch:", anomaly_mask)

