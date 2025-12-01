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
import os
import networkx as nx

# np.random.seed(42)
# torch.random.manual_seed(42)

class Correction_Module_dense(nn.Module):
    def __init__(self, k=4):
        super(Correction_Module_dense, self).__init__()
        self.mean_grad = {}
        self.var_grad = {}
        self.num_updates = {}
        self.k = k

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

    def forward(self, output, layer_name, replace_tensor=None):
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
        if replace_tensor is not None:
            new_output[mask] = replace_tensor[mask]
        else:
            new_output[mask] = 0

        return new_output

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
    flat_output = A.reshape(-1)

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
    flat_output = A.reshape(-1)

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

def row_checksum(A):
    m, n = A.shape
    row_sum = torch.sum(A, dim=1).view(m, 1)
    return torch.cat((A, row_sum), dim=1)
    
def col_checksum(B):
    m, n = B.shape
    col_sum = torch.sum(B, dim=0).view(1, n)
    return torch.cat((B, col_sum), dim=0)

def checksum_verify(left, right, epsilon=1e-4, inject=False, error_rate=1e-3, is_pruned=False):
    """
    Performs ABFT-style checksum verification for left @ right
    Returns (output, mask)
    """
    left_cs = col_checksum(left)
    right_cs = row_checksum(right)

    if is_pruned:
        zeros_mask = torch.isclose(right_cs, torch.zeros_like(right_cs))

    if inject:
        right_cs, _ = flip_bits(right_cs, error_rate=error_rate)
        if is_pruned:
            right_cs[zeros_mask] = 0

    prod_cs = left_cs @ right_cs

    # if inject:
    #     prod_cs, _ = flip_bits(prod_cs, error_rate=error_rate)

    output = prod_cs[:-1, :-1]
    row_sum_check = prod_cs[:-1, -1]
    col_sum_check = prod_cs[-1, :-1]

    col_sum = torch.sum(output, dim=0)
    row_sum = torch.sum(output, dim=1)

    col_diff = torch.abs(col_sum - col_sum_check)
    row_diff = torch.abs(row_sum - row_sum_check)

    row_mask = row_diff > epsilon     # shape [m]
    col_mask = col_diff > epsilon     # shape [n]
    
    mask = row_mask.unsqueeze(1) & col_mask.unsqueeze(0)
    # print(mask)

    return output, mask

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

            A = self.A
            if inject:
                A, _ = flip_bits(A, error_rate=error_rate)

            x_t = state @ A.T + u_t @ self.B.T

            # if inject:
            #     x_t, _ = flip_bits(x_t.detach(), error_rate=error_rate)

            state = x_t
            y_t = state @ self.C.T
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        activation = self.activation(y)
        norm = self.norm(activation)
        return torch.nan_to_num(norm, nan=0.0, posinf=1e6, neginf=-1e6)

    def single_step(self, state, u, inject=False, error_rate=1e-3):
        if state is None:
            state = torch.zeros(u.size(0), self.d_state, device=u.device)
        x_next = state @ self.A.T + u @ self.B.T
        if inject:
            x_next, _ = flip_bits(x_next.detach(), error_rate=error_rate)
        y = x_next @ self.C.T
        return x_next, y
    
    def hidden_update(self, x, u, inject=False, error_rate=1e-3):
        A, B = self.A, self.B
        
        left_term, left_mask = checksum_verify(x, A.T, inject=inject, error_rate=error_rate)
        right_term, right_mask = checksum_verify(u, B.T)

        x_next = left_term + right_term
        mask = left_mask | right_mask

        # if inject:
        #     A, _ = flip_bits(A, error_rate=error_rate)

        # left_term = x @ A.T
        # right_term = u @ B.T
        # x_next = left_term + right_term

        # if inject:
        #     x_next, _ = flip_bits(x_next, error_rate=error_rate)

        return x_next, mask
    
    def get_output(self, state):
        return state @ self.C.T

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
            if inject:
                x, _ = flip_bits(x, error_rate=error_rate)
            if correct:
                # detect if there is an error
                x, detected = self.corr_ssm(x, layer_name="ssm_output")
        x = self.pool(x).squeeze(-1)    # (B, d_model)

        return self.fc(x), detected, activations

def dequantize_tensor(q_tensor, scale, zero_point=0):
    return scale * (q_tensor - zero_point)

def to_float8(x, exp_bits=4, mant_bits=3):
    # compute parameters
    max_exp = 2**(exp_bits - 1) - 1
    min_exp = -max_exp
    scale = 2.0 ** mant_bits

    # approximate FP8 rounding by quantizing mantissa
    sign = torch.sign(x)
    x_abs = torch.abs(x)

    exp = torch.floor(torch.log2(x_abs + 1e-8))
    exp = torch.clamp(exp, min_exp, max_exp)
    mant = x_abs / (2 ** exp) - 1
    mant_q = torch.round(mant * scale) / scale

    return sign * (1 + mant_q) * (2 ** exp)

class QuantizedSSM(nn.Module):
    def __init__(self, A, B, C, int_bits=4, frac_bits=3):
        super().__init__()
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
        self.int_bits = int_bits
        self.frac_bits = frac_bits

    def forward(self, x, u):
        x_next, _ = self.hidden_update(x, u)
        y = self.get_output(x_next)
        return x_next, y
    
    def hidden_update(self, x, u, inject=False, error_rate=1e-3):
        A = to_float8(self.A, exp_bits=self.int_bits, mant_bits=self.frac_bits)
        B = to_float8(self.B, exp_bits=self.int_bits, mant_bits=self.frac_bits)

        left_term, left_mask = checksum_verify(x, A.T, inject=inject, error_rate=error_rate, is_pruned=True)
        right_term, right_mask = checksum_verify(u, B.T)

        x_next = left_term + right_term
        mask = left_mask | right_mask
        
        # if inject:
        #     x_next, _ = flip_bits(x_next.detach(), error_rate=error_rate)

        return to_float8(x_next, exp_bits=self.int_bits, mant_bits=self.frac_bits), mask

    def get_output(self, x):
        C = to_float8(self.C, exp_bits=self.int_bits, mant_bits=self.frac_bits)
        res = x @ C.T
        return to_float8(res, exp_bits=self.int_bits, mant_bits=self.frac_bits)
    
def prune_tensor(tensor, amount=0.3):
    k = int(amount * tensor.numel())
    threshold = tensor.abs().view(-1).kthvalue(k).values
    mask = (tensor.abs() >= threshold).float()
    return tensor * mask

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
        return model, dataset

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

    return model, dataset

import time
import statistics as stats

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, dataset = train_model(epochs=5, device=device)

    # Split the training dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)

    # # collect gradients
    # model.eval()
    # correct, total = 0, 0
    # with torch.no_grad():
    #     for x, y in tqdm(test_loader, desc=f"Collecting gradients", leave=False):
    #         x, y = x.to(device), y.to(device)
    #         logits = model(x, compute_grad=True)[0]
    #         preds = logits.argmax(dim=1)
    #         correct += (preds == y).sum().item()
    #         total += y.size(0)

    # test_acc = correct / total
    # print(f"Nominal Acc={test_acc:.4f}")

    # # faulty, no correction
    # model.eval()
    # correct, total = 0, 0
    # with torch.no_grad():
    #     for x, y in tqdm(test_loader, desc=f"Testing faulty, no correction", leave=False):
    #         x, y = x.to(device), y.to(device)
    #         logits = model(x, error_rate=5e-5, inject=True)[0]
    #         preds = logits.argmax(dim=1)
    #         correct += (preds == y).sum().item()
    #         total += y.size(0)

    # test_acc = correct / total
    # print(f"Faulty (no correction) Acc={test_acc:.4f}")

    # # faulty, with correction
    # model.eval()
    # correct, total = 0, 0
    # with torch.no_grad():
    #     for x, y in tqdm(test_loader, desc=f"Testing faulty, with correction", leave=False):
    #         x, y = x.to(device), y.to(device)
    #         logits = model(x, error_rate=5e-5, inject=True, correct=True)[0]
    #         preds = logits.argmax(dim=1)
    #         correct += (preds == y).sum().item()
    #         total += y.size(0)

    # test_acc = correct / total
    # print(f"Faulty (with correction) Acc={test_acc:.4f}")

    quantized_ssm = None
    A_q = model.ssm.A.data.clone()
    B_q = model.ssm.B.data.clone()
    C_q = model.ssm.C.data.clone()

    if not os.path.exists("quantized_ssm.pth"):

        for j in range(4):
            if j != 0:
                # desired sparsity for iteration j
                sparsity = 0.2 + j * 0.1
                A_q = prune_tensor(A_q, amount=sparsity)
                B_q = prune_tensor(B_q, amount=sparsity)
                C_q = prune_tensor(C_q, amount=sparsity)

                A_mask = (A_q != 0).float()
                B_mask = (B_q != 0).float()
                C_mask = (C_q != 0).float()

            child = QuantizedSSM(A_q, B_q, C_q).to(device)

            optimizer = torch.optim.Adam(child.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            num_epochs = 20
            for epoch in range(num_epochs):
                child.train()
                model.eval()
                for x, y in tqdm(train_loader, desc=f"Quantied SSM Training Epoch {epoch+1}/{num_epochs}", leave=False):
                    x, y = x.to(device), y.to(device)
                
                    x = x.squeeze(1)                # (B, 28, 28)
                    x = model.input_proj(x)          # (B, 28, d_model)

                    B, L, D = x.shape
                    state = torch.zeros(B, child.A.size(0), device=device)
                    parent_state = torch.zeros(B, model.ssm.d_state, device=device)

                    loss = 0
                    for t in range(L):
                        u_t = x[:, t, :]
                        state, y_t = child(state, u_t)
                        with torch.no_grad():
                            parent_state, parent_y_t = model.ssm.single_step(parent_state, u_t)

                        loss += criterion(state, parent_state.detach()) + criterion(y_t, parent_y_t.detach())

                    loss /= L
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # reapply sparsity masks
                    if j != 0:
                        with torch.no_grad():
                            child.A.mul_(A_mask)
                            child.B.mul_(B_mask)
                            child.C.mul_(C_mask)

            # current sparsity
            total_params = child.A.numel() + child.B.numel() + child.C.numel()
            zero_params = (child.A == 0).sum().item() + (child.B == 0).sum().item() + (child.C == 0).sum().item()
            sparsity = zero_params / total_params
            print(f"Quantized SSM Model Sparsity: {sparsity*100:.2f}%")

            model.eval()
            child.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in tqdm(test_loader, desc="Testing Quantized SSM Model", leave=False):
                    x, y = x.to(device), y.to(device)
                    x = x.squeeze(1)                # (B, 28, 28)
                    x = model.input_proj(x)          # (B, 28, d_model)
                    B, L, D = x.shape
                    state = torch.zeros(B, child.A.size(0), device=device)
                    outputs = []
                    for t in range(L):
                        u_t = x[:, t, :]
                        state, y_t = child(state, u_t)
                        outputs.append(y_t)
                    output = torch.stack(outputs, dim=1)
                    # output = output.to(torch.float32)
                    output = model.ssm.activation(output)
                    output = model.ssm.norm(output)
                    x = output.transpose(1, 2).contiguous()      # (B, d_model, 28)
                    x = model.pool(x).squeeze(-1)    #
                    logits = model.fc(x)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            test_acc = correct / total
            print(f"Quantized SSM Model Test Acc={test_acc:.4f}")
        
        quantized_ssm = child

        # save the quantized ssm
        torch.save(child.state_dict(), "quantized_ssm.pth")
    else:
        print("Quantized SSM model already exists at quantized_ssm.pth")
        A_q = model.ssm.A.data.clone()
        B_q = model.ssm.B.data.clone()
        C_q = model.ssm.C.data.clone()
        quantized_ssm = QuantizedSSM(A_q, B_q, C_q).to(device)
        quantized_ssm.load_state_dict(torch.load("quantized_ssm.pth"))
    
    # test the quantized model and collect gradients
    grad_collector = Correction_Module_dense(k=4)

    model.eval()
    quantized_ssm.eval()
    correct, total = 0, 0
    diff_vals = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing Quantized SSM Model", leave=False):
            x, y = x.to(device), y.to(device)
            x = x.squeeze(1)                # (B, 28, 28)
            x = model.input_proj(x)          # (B, 28, d_model)
            B, L, D = x.shape
            state = torch.zeros(B, quantized_ssm.A.size(0), device=device)
            model_state = torch.zeros(B, model.ssm.d_state, device=device)
            outputs = []
            for t in range(L):
                u_t = x[:, t, :]
                state, y_t = quantized_ssm(state, u_t)
                model_state, _ = model.ssm.hidden_update(model_state, u_t)

                # grad_collector.compute_grad(model_state, f"ssm_state_{t}")
                diff_vals.append(torch.abs(model_state - state))

                outputs.append(y_t)
            output = torch.stack(outputs, dim=1)
            # output = output.to(torch.float32)
            output = model.ssm.activation(output)
            output = model.ssm.norm(output)
            x = output.transpose(1, 2).contiguous()      # (B, d_model, 28)
            x = model.pool(x).squeeze(-1)    #
            logits = model.fc(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    print(f"Quantized SSM Model Test Acc={test_acc:.4f}")

    diff_vals = torch.stack(diff_vals, dim=0)
    mu_diff = torch.mean(diff_vals).item()
    std_diff = torch.std(diff_vals).item()
    print(f"mean and std dev of delta: mu={mu_diff:.4f}, sigma={std_diff:.4f}")

    k = 5
    execution_times = []

    # use it to correct the original ssm when faulty
    model.eval()
    quantized_ssm.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing SSM Model", leave=False):
            x, y = x.to(device), y.to(device)
            x = x.squeeze(1)                # (B, 28, 28)
            x = model.input_proj(x)          # (B, 28, d_model)
            B, L, D = x.shape
            model_state = torch.zeros(B, model.ssm.d_state, device=device)
            quantized_state = torch.zeros(B, quantized_ssm.A.size(0), device=device)
            outputs = []
            for t in range(L):
                start = time.perf_counter()

                u_t = x[:, t, :]
                quantized_state, quant_mask = quantized_ssm.hidden_update(quantized_state, u_t, inject=True, error_rate=5e-5)
                model_state, _ = model.ssm.hidden_update(model_state, u_t, inject=True, error_rate=5e-5)

                model_state = nan_checker(model_state)
                quantized_state = nan_checker(quantized_state)
                quantized_state[quant_mask] = 0

                # model_state = grad_collector(model_state, f"ssm_state_{t}")
                diff = torch.abs(model_state - quantized_state)
                mask = torch.abs(diff - mu_diff) > k * std_diff
                model_state[mask] = quantized_state[mask]

                y_t = model.ssm.get_output(model_state)
                outputs.append(y_t)

                end = time.perf_counter()
                exec_time = end - start
                execution_times.append(exec_time)

            output = torch.stack(outputs, dim=1)
            output = model.ssm.activation(output)
            output = model.ssm.norm(output)
            x = output.transpose(1, 2).contiguous()      # (B, d_model, 28)
            x = model.pool(x).squeeze(-1)    #
            logits = model.fc(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    test_acc = correct / total
    print(f"Corrected SSM Model Test Acc={test_acc:.4f}")
    median_exec_time = stats.median(execution_times) * 1000
    mean_exec_time = stats.mean(execution_times) * 1000
    stdev_exec_time = stats.stdev(execution_times) * 1000
    print(f"Avg execution time={mean_exec_time:.4f} ms")
    print(f"Median execution time={median_exec_time:.4f} ms")