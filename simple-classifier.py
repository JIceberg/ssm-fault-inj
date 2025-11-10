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
from matplotlib import pyplot as plt
import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, kstest
import numpy as np
import networkx as nx

# np.random.seed(42)
# torch.random.manual_seed(42)

class Correction_Module_ssm(nn.Module):
    def __init__(self):
        super(Correction_Module_ssm, self).__init__()
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}
        self.k = 6  # threshold multiplier

    def get_gradient(self, x):
        """
        x: (B, N, K)
        Apply a 1D convolution over the last dimension (K).
        """
        B, N, K = x.shape
        conv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=3,
            padding='same', padding_mode='circular', bias=False
        )
        with torch.no_grad():
            conv.weight[0, 0, :] = torch.tensor([-1, 1, 0], dtype=torch.float)
        conv = conv.to(x.device)

        # apply conv per neuron independently
        grad = conv(x.view(B * N, 1, K))  # (B*N, 1, K)
        return grad.view(B, N, K)  # reshape back

    def compute_grad(self, x, layer_name):
        """
        Updates running mean and variance of gradient magnitudes across batch.
        x: (B, N, K)
        """
        if layer_name not in self.num_updates:
            self.num_updates[layer_name] = 0
        self.num_updates[layer_name] += 1

        x = nan_checker(x)
        grad_Y = self.get_gradient(x)

        # compute mean/var across batch and k dimension
        mean = grad_Y.mean(dim=(0, 2)).cpu().detach().numpy()  # shape: (N,)
        var = grad_Y.var(dim=(0, 2)).cpu().detach().numpy()    # shape: (N,)

        if layer_name not in self.mean_grad:
            self.mean_grad[layer_name] = mean
            self.var_grad[layer_name] = var
        else:
            n = self.num_updates[layer_name]
            self.mean_grad[layer_name] += (mean - self.mean_grad[layer_name]) / n
            self.var_grad[layer_name] += (var - self.var_grad[layer_name]) / n

    def forward(self, output, layer_name):
        """
        Applies correction based on gradient outlier detection.
        output: (B, N, K)
        """
        if layer_name not in self.mean_grad:
            return output

        B, N, K = output.shape
        output = nan_checker(output)

        std_grad = np.sqrt(self.var_grad[layer_name])
        mean_grad_tensor = torch.tensor(self.mean_grad[layer_name], device=output.device).view(1, N, 1)
        std_grad_tensor = torch.tensor(std_grad, device=output.device).view(1, N, 1)

        lower_bound = mean_grad_tensor - self.k * std_grad_tensor
        upper_bound = mean_grad_tensor + self.k * std_grad_tensor

        grad_Y = self.get_gradient(output)
        mask = (grad_Y < lower_bound) | (grad_Y > upper_bound)

        new_output = output.clone()
        new_output[mask] = 0

        return new_output, mask.sum().item()

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


def flip_bits_fixed(A, num_flips=1, clamp_val=1e6):
    """
    Flip a random bit in `num_flips` random elements of tensor A.

    Args:
        A (torch.Tensor): Input tensor.
        num_flips (int): Number of elements to flip a bit in.
        clamp_val (float): Value to clamp to avoid inf/nan.
    
    Returns:
        modified_output (torch.Tensor): Tensor after bit flips.
        flipped_indices (np.ndarray): Indices of elements that were flipped.
    """
    # Flatten tensor
    flat_output = A.view(-1)

    # Convert to int32 representation
    float_bits = flat_output.to(torch.float32).cpu().numpy().view(np.uint32)

    # Select random elements to flip
    num_elements = flat_output.numel()
    if num_flips > num_elements:
        raise ValueError("num_flips cannot exceed total number of elements")
    flip_indices = np.random.choice(num_elements, size=num_flips, replace=False)

    # For each selected element, flip a random bit
    random_bits = np.random.randint(0, 32, size=num_flips, dtype=np.uint32)
    float_bits[flip_indices] ^= (1 << random_bits)

    # Convert back to tensor
    modified_output = torch.tensor(float_bits.view(np.float32), dtype=torch.float32, device=A.device).view(A.shape)

    # Clamp and replace NaNs
    modified_output = torch.nan_to_num(modified_output, nan=0.0, posinf=clamp_val, neginf=-clamp_val)

    return modified_output, flip_indices


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

            if inject:
                x_t, _ = flip_bits(x_t.detach(), error_rate=error_rate)

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
        self.corr_ssm = Correction_Module_ssm()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, error_rate=1e-4, compute_grad=False, inject=False, correct=False):
        x = x.squeeze(1)                # (B, 28, 28)
        x = self.input_proj(x)          # (B, 28, d_model)
        x = self.ssm(x, error_rate=error_rate, inject=inject)
        
        B, L, D = x.shape
        activations = x.reshape(B, L * D).detach()

        x = x.transpose(1, 2).contiguous()      # (B, d_model, 28)

        detected = 0

        if compute_grad:
            self.corr_ssm.compute_grad(x, layer_name="ssm_output")
        else:
            # if inject:
            #     # x, _ = flip_bits_fixed(x, num_flips=1)
            if correct:
                # detect if there is an error
                x, detected = self.corr_ssm(x, layer_name="ssm_output")
        x = self.pool(x).squeeze(-1)    # (B, d_model)

        return self.fc(x), detected, activations


# ---------- Training Script ----------
def train_model(epochs=5, batch_size=64, lr=1e-3, device="cuda"):
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)

    train_size = int(0.5 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split the training dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model, optimizer, loss
    model = SSMClassifier().to(device)

    # if model already is saved to path mnist_model.pth, load it
    import os
    if os.path.exists("mnist_model.pth"):
        model.load_state_dict(torch.load("mnist_model.pth"))
        print("Loaded pre-trained model from mnist_model.pth")
        return model, test_loader

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)[0]
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
                logits = model(x)[0]
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        test_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "mnist_model.pth")

    return model, test_loader

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, test_loader = train_model(epochs=5, device=device)

    