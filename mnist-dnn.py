import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

# -----------------------------
# 1. Define Models
# -----------------------------
class NeuronFilter(nn.Module):
    def __init__(self, k=5, tau=3.0, eps=1e-6):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.tau = tau
        self.eps = eps

    def forward(self, a):
        """
        a: [B, N] activation window
        """
        a = a.unsqueeze(1)  # [B, 1, N]
        return self.conv(a).squeeze(1)  # [B, N]
    
    def correct(self, A):
        """
        A: activations, shape (B, N)
        returns: stabilized activations of same shape
        """
        # Reshape for Conv1d
        x = A.unsqueeze(1)  # (B, 1, N)
        Y = self.conv(x)    # (B, 1, N')
        
        # Adjust Y to same length if conv reduces size
        if Y.shape[-1] != A.shape[-1]:
            pad_len = A.shape[-1] - Y.shape[-1]
            Y = torch.nn.functional.pad(Y, (pad_len // 2, pad_len - pad_len // 2))

        # Compute robust statistics along batch dimension
        mu = A.median(dim=0, keepdim=True).values
        mad = (A - mu).abs().median(dim=0, keepdim=True).values
        mad = torch.clamp(mad, min=self.eps)

        # Compute robust z-score
        z = (A - mu) / (1.4826 * mad)
        mask = (z.abs() > self.tau).float()  # 1 where outlier

        # Replace outliers with filtered values
        A_out = (1 - mask) * A + mask * Y.squeeze(1)
        return A_out


class MNIST_DNN(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dims=[256, 128], num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x, inject=False, error_rate=1e-3, filter=None):
        x = x.view(x.size(0), -1)
        h1 = self.fc1(x)
        if inject:
            h1, _ = flip_bits(h1, error_rate=error_rate)
            if filter is not None:
                h1 = filter.correct(h1)

        x = F.relu(h1)
        h2 = F.relu(self.fc2(x))
        out = self.fc3(h2)
        return out, h1.detach()


# -----------------------------
# 2. Prepare Data
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_DNN().to(device)
filter_net = NeuronFilter(k=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3)

# -----------------------------
# 3. Train or Load Base Model
# -----------------------------
K = 0.01  # regularization weight
if not os.path.exists("mnist_dnn.pth"):
    num_epochs = 10
    loss_per_epoch = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs, h1 = model(images)
            # with torch.no_grad():
            #     Y = filter_net(h1)
            loss = criterion(outputs, labels) # + K * Y.std(dim=0).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        loss_per_epoch.append(epoch_loss)

    # plot loss curve
    plt.plot(range(1, num_epochs + 1), loss_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid()
    plt.savefig("mnist_dnn_loss_curve.png")
    
    torch.save(model.state_dict(), "mnist_dnn.pth")
    torch.save(filter_net.state_dict(), "mnist_filter.pth")
else:
    model.load_state_dict(torch.load("mnist_dnn.pth", map_location=device))
    filter_net.load_state_dict(torch.load("mnist_filter.pth", map_location=device))

# test base model accuracy
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing Base Model", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Base model accuracy on test set: {100 * correct / total:.2f}%")

# test with bit flips
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing Faulty Model", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images, inject=True, error_rate=1e-1)
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
print(f"Faulty model accuracy on test set: {100 * correct / total:.2f}%")

import random
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, kstest
import numpy as np
import networkx as nx

collected_acts = []

model.eval()
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="collecting activations"):
        x, y = x.to(device), y.to(device)
        logits, activations = model(x)

        activations = activations.cpu()

        collected_acts.append(activations)

all_acts = torch.cat(collected_acts, dim=0)  # shape [N_samples, N_neurons]

# Convert to numpy for convenience
A = all_acts.numpy()

# Compute correlation matrix: shape [N_neurons, N_neurons]
corr_matrix = np.corrcoef(A, rowvar=False)

print("Correlation matrix shape:", corr_matrix.shape)

# plt.figure(figsize=(10,8))
# sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1)
# plt.title("Correlation Matrix of Neuron Activations (h1 Layer)")
# plt.xlabel("Neuron index")
# plt.ylabel("Neuron index")
# plt.show()

cov_matrix = np.cov(A, rowvar=False)  # shape [N_neurons, N_neurons]

num_neurons = A.shape[1]

node_stats = {}
for i in range(num_neurons):
    mu_i, sigma_i = norm.fit(A[:, i])  # fit N(μ, σ)
    node_stats[i] = {'mu': mu_i, 'sigma': sigma_i}

threshold = 0.5  # correlation threshold
N = num_neurons
G_stats = nx.Graph()
G_stats.add_nodes_from(range(N))

# Add mean/std to each node
for n in range(N):
    G_stats.nodes[n]['mu'] = node_stats[n]['mu']
    G_stats.nodes[n]['sigma'] = node_stats[n]['sigma']

# Add edges where |corr| ≥ threshold, weighted by covariance
for i in range(N):
    for j in range(i + 1, N):
        if corr_matrix[i, j] >= threshold:
            G_stats.add_edge(i, j, weight=cov_matrix[i, j])


print(f"Graph with {G_stats.number_of_nodes()} nodes and {G_stats.number_of_edges()} edges.")

avg_deltas = {}       # dictionary to store Δ_ij for each edge (i,j)

for i, j in G_stats.edges():
    # compute mean difference across all samples
    delta = np.mean(A[:, j] - A[:, i])
    avg_deltas[(i, j)] = delta

for (i, j), delta in avg_deltas.items():
    G_stats[i][j]['avg_delta'] = delta

import random

# create test loader with batch size 1 for simplicity
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

k, l = 4, 6

model.eval()
correct = 0
total = 0
for images, labels in tqdm(test_loader, desc=f"Testing Model", leave=False):
    images, labels = images.to(device), labels.to(device)
    
    x = images.view(images.size(0), -1)
    h1 = model.fc1(x)

    original_h1 = h1.clone()

    # select random neuron idx
    random_idx = torch.randint(0, h1.size(1), (1,)).item()
    # flip a random bit in h1[:, random_idx]
    bit_to_flip = 1 << random.randint(0, 15)  # assuming 16-bit representation
    h1_int = h1[:, random_idx].to(torch.int32)
    h1_int_flipped = h1_int ^ bit_to_flip
    h1[:, random_idx] = h1_int_flipped.to(h1.dtype)

    erroneous_idx = []
    for i in range(h1.size(1)):
        neighbors = list(G_stats.neighbors(i))
        # if len(neighbors) == 0:
        # get node i stats
        mu_i = node_stats[i]['mu']
        std_i = node_stats[i]['sigma']
        # check if h1[:, i] is l std dev away from mu_i
        mask = (torch.abs(h1[:, i] - mu_i) > l * std_i)
        # h1[mask, i] = torch.tensor(mu_i, dtype=h1.dtype, device=h1.device)
        if mask.any():
            erroneous_idx.append(i)

    # print("detected idx vs actual idx:", erroneous_idx, random_idx)

    # print("errored output vs original output", h1[:, random_idx], original_h1[:, random_idx])

    for a_idx in erroneous_idx:
        neighbors = list(G_stats.neighbors(a_idx))
        if len(neighbors) == 0:
            print(f"Neuron a = {a_idx} has no neighbors, using mean value.")
            h1[:, a_idx] = torch.tensor(node_stats[a_idx]['mu'], dtype=h1.dtype, device=h1.device).repeat(h1.size(0))
        else:
            cov_ax = np.array([cov_matrix[a_idx, j] for j in neighbors])             # [1 x k]
            cov_xx = np.array([[cov_matrix[i, j] for j in neighbors] for i in neighbors])  # [k x k]

            mu_a = node_stats[a_idx]['mu']
            mu_x = np.array([node_stats[j]['mu'] for j in neighbors])

            x_vals = h1[:, neighbors].detach().cpu().numpy()  # activations of neighbors
            x_centered = x_vals - mu_x

            val = mu_a + cov_ax @ np.linalg.inv(cov_xx) @ x_centered.T  # [1 x B]
            h1[:, a_idx] = torch.tensor(val, dtype=h1.dtype, device=h1.device).squeeze()

        # print("corrected output vs original output:", h1[:, a_idx], original_h1[:, a_idx])

    x = F.relu(h1)
    h2 = F.relu(model.fc2(x))
    logits = model.fc3(h2)

    preds = torch.argmax(logits, dim=1)
    correct += (preds == labels).sum().item()
    total += labels.size(0)
accuracy = correct / total
print(f"Correct accuracy: {accuracy*100:.2f}%")

# # N = cov_matrix.shape[0]

# # # sample_idx = random.randint(0, all_acts.shape[0]-1)

# # for sample_idx in range(all_acts.shape[0]):
# #     num_correct = 0
# #     num_total = 0
# #     for a_idx in tqdm(range(N), desc=f"Predicting Neurons for Sample {sample_idx}", leave=False):
# #         neighbors = list(G_stats.neighbors(a_idx))  # neighbors based on corr threshold

# #         if len(neighbors) == 0:
# #             # print(f"Neuron a = {a_idx} has no neighbors, skipping.")
# #             continue

# #         # print(f"Neuron a = {a_idx}, Neighbors = {neighbors}")

# #         # Extract the relevant submatrices/vectors
# #         cov_ax = np.array([cov_matrix[a_idx, j] for j in neighbors])             # [1 x k]
# #         cov_xx = np.array([[cov_matrix[i, j] for j in neighbors] for i in neighbors])  # [k x k]

# #         # Extract means
# #         mu_a = node_stats[a_idx]['mu']
# #         std_a = node_stats[a_idx]['sigma']
# #         mu_x = np.array([node_stats[j]['mu'] for j in neighbors])

# #         # Example: pick a single sample's activations to condition on
# #         x_vals = all_acts[sample_idx, neighbors].numpy()  # activations of neighbors
# #         x_centered = x_vals - mu_x

# #         # Compute conditional mean estimate for neuron a
# #         val = mu_a + cov_ax @ np.linalg.inv(cov_xx) @ x_centered

# #         # determine if this is within 1 std of actual activation
# #         actual_a = all_acts[sample_idx, a_idx].item()
# #         diff = abs(val - actual_a)
# #         within_std = diff <= (std_a / 2)
# #         if within_std:
# #             num_correct += 1
# #         num_total += 1

# #         # print(f"Neuron a={a_idx}, Sample={sample_idx}, Predicted E[a|neighbors]={val:.4f}, Actual a={actual_a:.4f}, Diff={diff:.4f}, Within 1 std: {within_std}")

# #         # print(f"Predicted E[a|neighbors] = {val:.4f}")
# #         # print(f"Actual a activation (sample {sample_idx}) = {all_acts[sample_idx, a_idx]:.4f}")
# #         # print(f"Difference = {abs(val - all_acts[sample_idx, a_idx]):.4f} vs std {std_a:.4f}")

# #     accuracy = num_correct / num_total if num_total > 0 else 0.0
# #     print(f"Overall accuracy of predicting neuron activations within 0.5 std: {accuracy*100:.2f}% ({num_correct}/{num_total})")

# # # Pick a random neuron index
# # hidden_dim = collected_data[0][0].shape[0]
# # j = random.randint(0, hidden_dim - 2)
# # print(f"Plotting activation distribution for neuron #{j}")

# # # Collect activations for that neuron across all samples
# # neuron_values = [acts[j].item() for acts, _, _ in collected_data]
# # neuron_values = np.array(neuron_values)

# # # Fit a normal distribution to the data
# # mu_hat, sigma_hat = norm.fit(neuron_values)

# # print(f"Estimated μ = {mu_hat:.4f}, σ = {sigma_hat:.4f}")

# # # Plot histogram and KDE
# # plt.figure(figsize=(7, 5))
# # sns.histplot(neuron_values, bins=30, kde=False, color='skyblue', stat='density', label='Empirical')
# # x = np.linspace(neuron_values.min(), neuron_values.max(), 200)
# # plt.plot(x, norm.pdf(x, mu_hat, sigma_hat), 'r-', lw=2, label=f'N({mu_hat:.2f}, {sigma_hat:.2f})')
# # plt.title(f"Activation Distribution for Neuron #{j}")
# # plt.xlabel("Activation Value")
# # plt.ylabel("Frequency")
# # plt.grid(True)
# # plt.show()

# # # Pick a random adjacent pair (assumes collected_data items are (acts, mu_in, var_in))
# # print(f"Adjacent pair: neurons {j} and {j+1}")

# # # Collect differences across the dataset
# # diff_vals = []
# # for acts, _, _ in collected_data:
# #     diff_vals.append((acts[j+1] - acts[j]).item())
# # diff_vals = np.array(diff_vals)

# # # Fit a Normal(μ̂, σ̂) and test normality
# # mu_hat, sigma_hat = norm.fit(diff_vals)
# # D, p = kstest(diff_vals, 'norm', args=(mu_hat, sigma_hat))
# # print(f"Δ mean={mu_hat:.4f}, std={sigma_hat:.4f}, KS={D:.4f}, p={p:.3g}")

# # # Plot histogram + fitted PDF
# # plt.figure(figsize=(7,5))
# # counts, bins, _ = plt.hist(diff_vals, bins=40, density=True, alpha=0.6)
# # x = np.linspace(bins[0], bins[-1], 300)
# # plt.plot(x, norm.pdf(x, mu_hat, sigma_hat), lw=2)
# # plt.title(f"Distribution of a[j+1] - a[j] for j={j}")
# # plt.xlabel("Δ activation")
# # plt.ylabel("Density")
# # plt.grid(True); plt.show()
