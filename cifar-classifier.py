import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import math
import argparse
import os
import numpy as np
import time
import statistics as stats
import torch.nn.utils.prune as prune
from matplotlib import pyplot as plt
import json

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
    modified_output = torch.nan_to_num(modified_output, nan=0.0, posinf=0.0, neginf=0.0)
    
    return modified_output, does_flip

def to_float8(x, exp_bits=4, mant_bits=3):
    max_exp = 2**(exp_bits - 1) - 1
    min_exp = -max_exp
    scale = 2.0 ** mant_bits

    sign = torch.sign(x)
    x_abs = torch.abs(x)

    exp = torch.floor(torch.log2(x_abs + 1e-8))
    exp = torch.clamp(exp, min_exp, max_exp)
    mant = x_abs / (2 ** exp) - 1
    mant_q = torch.round(mant * scale) / scale
    return sign * (1 + mant_q) * (2 ** exp)

def dequantize_tensor(q_tensor, scale, zero_point=0):
    return scale * (q_tensor - zero_point)

def row_checksum(A):
    m, n = A.shape
    row_sum = torch.sum(A, dim=1).view(m, 1)
    return torch.cat((A, row_sum), dim=1)
    
def col_checksum(B):
    m, n = B.shape
    col_sum = torch.sum(B, dim=0).view(1, n)
    return torch.cat((B, col_sum), dim=0)

def checksum_verify(left, right, epsilon=1e-4, weight_inject=False, out_inject=False, error_rate=1e-3, is_pruned=False):
    """
    Performs ABFT-style checksum verification for left @ right
    Returns (output, mask)
    """
    left_cs = col_checksum(left)
    right_cs = row_checksum(right)

    if is_pruned:
        zeros_mask = torch.isclose(right_cs, torch.zeros_like(right_cs))

    if weight_inject:
        right_cs, _ = flip_bits(right_cs, error_rate=error_rate)
        if is_pruned:
            right_cs[zeros_mask] = 0

    prod_cs = left_cs @ right_cs

    if out_inject:
        prod_cs, _ = flip_bits(prod_cs, error_rate=error_rate)

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

def prune_tensor(tensor, amount=0.3):
    k = int(amount * tensor.numel())
    threshold = tensor.abs().view(-1).kthvalue(k).values
    mask = (tensor.abs() >= threshold).float()
    return tensor * mask, mask

# ---------------------------
# Simple SSM / S4-like layer
# ---------------------------
class SimpleSSM(nn.Module):
    """
    A simplified state-space model layer:
      x_t = A * x_{t-1} + B * u_t
      y_t = C * x_t + D * u_t

    A is diagonal (learned vector) for efficiency. Works on input shaped (B, input_dim, L)
    and returns output shaped (B, output_dim, L).
    This is simple, stable if A values are initialized appropriately, but is not the
    full S4 kernel/diagonalization optimization.
    """
    def __init__(self, input_dim, state_dim, output_dim, init_scale=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        # diagonal entries of A (we parametrize in unconstrained space and use tanh to keep them bounded)
        self.A_unconstrained = nn.Parameter(torch.randn(state_dim) * 0.1)
        # B: (state_dim, input_dim)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * init_scale / math.sqrt(input_dim))
        # C: (output_dim, state_dim)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * init_scale / math.sqrt(state_dim))
        # D: (output_dim, input_dim)
        self.D = nn.Parameter(torch.zeros(output_dim, input_dim))

        # Optional initial state bias
        self.h0 = nn.Parameter(torch.zeros(state_dim))

    def forward(self, u):
        # u shape: (B, input_dim, L)
        Bz, input_dim, L = u.shape
        assert input_dim == self.input_dim

        # convert to (L, B, input_dim) for time-major loop
        u_t = u.permute(2, 0, 1)  # (L, B, input_dim)

        # state: (B, state_dim)
        state = self.h0.unsqueeze(0).expand(Bz, -1).contiguous()

        # A: get diagonal in a stable range (negative real part encourages stability)
        # Use tanh or -softplus to encourage negative values. We'll use -softplus to make entries negative.
        A_diag = -F.softplus(self.A_unconstrained)  # shape (state_dim,)

        outputs = []
        for t in range(L):
            ut = u_t[t]  # (B, input_dim)
            # state update: state = A * state + B @ u_t
            # A * state => elementwise multiply
            x_t = state * A_diag.unsqueeze(0) + (ut @ self.B.t())  # (B, state_dim)

            state = x_t

            # y_t = C @ state + D @ u_t
            y_t = state @ self.C.t() + (ut @ self.D.t())  # (B, output_dim)
            outputs.append(y_t)

        # stack => (L, B, output_dim) -> permute to (B, output_dim, L)
        Y = torch.stack(outputs, dim=0).permute(1, 2, 0).contiguous()
        return Y
    
    def single_step(self, state, u):
        state_next, _ = self.hidden_update(state, u)
        return state_next, self.get_output(state_next, u)
    
    def hidden_update(self, state, u, weight_inject=False, out_inject=False, error_rate=1e-3):
        A, B = self.A_unconstrained, self.B
        A_diag = -F.softplus(self.A_unconstrained)

        x_term = state * A_diag.unsqueeze(0)
        u_term, mask = checksum_verify(u, B.t(), weight_inject=weight_inject,
                                       out_inject=out_inject, error_rate=error_rate)

        state_next = x_term + u_term

        return state_next, mask

    def get_output(self, state, u):
        C, D = self.C, self.D
        y = state @ C.t() + (u @ D.t())
        return y
    
class QuantizedSimpleSSM(nn.Module):
    def __init__(self, A_diag, B, C, D, int_bits=4, mant_bits=3):
        super().__init__()
        self.A_diag = nn.Parameter(A_diag.clone())   # shape: (state_dim,)
        self.B = nn.Parameter(B.clone())             # (state_dim, input_dim)
        self.C = nn.Parameter(C.clone())             # (output_dim, state_dim)
        self.D = nn.Parameter(D.clone())             # (output_dim, input_dim)
        self.int_bits = int_bits
        self.frac_bits = mant_bits

    def quant(self, t):
        return to_float8(t, exp_bits=self.int_bits, mant_bits=self.frac_bits)

    def forward(self, u):
        Bz, input_dim, L = u.shape
        u_t = u.permute(2,0,1)

        state_dim = self.A_diag.shape[0]
        state = torch.zeros(Bz, state_dim, device=u.device)

        A_q = self.quant(self.A_diag).unsqueeze(0)  # (1,state_dim)
        B_q = self.quant(self.B)                    # (state_dim,input_dim)
        C_q = self.quant(self.C)
        D_q = self.quant(self.D)

        outputs = []
        for t in range(L):
            ut = u_t[t]

            # Diagonal ABFT update = safe elementwise multiply
            # A is diagonal → no matrix multiply needed
            state_next = state * A_q

            # ABFT for ut @ B.T
            right_term, maskB = checksum_verify(ut, B_q.t())
            state_next = state_next + right_term

            state = self.quant(state_next)

            y_t = state @ C_q.t() + (ut @ D_q.t())
            y_t = self.quant(y_t)
            outputs.append(y_t)

        Y = torch.stack(outputs, dim=0).permute(1,2,0)
        return Y
    
    def hidden_update(self, state, u, weight_inject=False, out_inject=False, error_rate=1e-3):
        A_diag = to_float8(self.A_diag, exp_bits=self.int_bits, mant_bits=self.frac_bits)
        B = to_float8(self.B, exp_bits=self.int_bits, mant_bits=self.frac_bits)

        state_next = state * A_diag.unsqueeze(0)
        u_term, mask = checksum_verify(u, B.t(), weight_inject=weight_inject,
                                       out_inject=out_inject, error_rate=error_rate, is_pruned=True)

        state_next = state_next + u_term

        return to_float8(state_next, exp_bits=self.int_bits, mant_bits=self.frac_bits), mask

    def get_output(self, state, u):
        C = to_float8(self.C, exp_bits=self.int_bits, mant_bits=self.frac_bits)
        D = to_float8(self.D, exp_bits=self.int_bits, mant_bits=self.frac_bits)
        y = state @ C.t() + (u @ D.t())
        return to_float8(y, exp_bits=self.int_bits, mant_bits=self.frac_bits)

# ---------------------------
# CIFAR-10 model using SSM
# ---------------------------
class CIFAR_SSM_Classifier(nn.Module):
    def __init__(self, ssm_state_dim=256, ssm_out_dim=256):
        super().__init__()
        # simple conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Reduce width to 1 so we have a sequence of length 32 (rows)
        # After stem: (B, 128, 32, 32)
        self.pool_width_to_1 = nn.AdaptiveAvgPool2d((32, 1))  # -> (B,128,32,1)

        # SSM operates on sequence length = 32, input_dim = 128
        self.ssm = SimpleSSM(input_dim=128, state_dim=ssm_state_dim, output_dim=ssm_out_dim)

        # classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # average over sequence dimension -> (B, ssm_out_dim, 1)
            nn.Flatten(),
            nn.Linear(ssm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

        self.corr_ssm = Correction_Module_ssm()

    def forward(self, x, inject=False, error_rate=1e-3, compute_grad=False, correct=False):
        # x: (B,3,32,32)
        z = self.stem(x)                     # (B,128,32,32)
        z = self.pool_width_to_1(z)          # (B,128,32,1)
        z = z.squeeze(-1)                    # (B,128,32) sequence: L=32, input_dim=128
        y = self.ssm(z, inject=inject, error_rate=error_rate)   # (B, ssm_out_dim, 32)

        activations = y.mean(axis=2).detach()

        if compute_grad:
            self.corr_ssm.compute_grad(y, layer_name="ssm_output")
        if inject:
            y, _ = flip_bits(y, error_rate=error_rate)
            if correct:
                y = self.corr_ssm(y, layer_name="ssm_output")
        out = self.head(y)                   # (B,10)
        # remove nans
        # out = nan_checker(out)
        return out, activations

# ---------------------------
# Training / utilities
# ---------------------------
def get_dataloaders(batch_size=128, num_workers=4, augment=True):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    train_transforms = []
    if augment:
        train_transforms += [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip()
        ]
    train_transforms += [T.ToTensor(), T.Normalize(mean, std)]
    train_transform = T.Compose(train_transforms)
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc='Train', leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=100*correct/total)
    return running_loss / total, 100.0 * correct / total

def evaluate(model, dataloader, device, criterion,
             inject=False, error_rate=1e-3, compute_grad=False, correct_error=False, desc='Eval'):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=desc, leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            # if inject:
            #     print(f"Evaluating with fault injection (error rate={error_rate}, correct={correct})")
            outputs, _ = model(images, inject=inject, error_rate=error_rate, compute_grad=compute_grad, correct=correct_error)
            # if inject:
            #     print(outputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=100*correct/total)
    return running_loss / total, 100.0 * correct / total

# ---------------------------
# Main: training loop
# ---------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Device:", device)

    trainloader, testloader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, augment=not args.no_augment)

    model = CIFAR_SSM_Classifier(ssm_state_dim=args.ssm_state_dim, ssm_out_dim=args.ssm_out_dim)
    model = model.to(device)

    # use AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, device, criterion)
        val_loss, val_acc = evaluate(model, testloader, device, criterion)
        scheduler.step()
        print(f"Train loss {train_loss:.4f} acc {train_acc:.2f} | Val loss {val_loss:.4f} acc {val_acc:.2f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = args.save_path or 'best_cifar_ssm.pth'
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'acc': best_acc}, save_path)
            print("Saved best model to", save_path)

    print("Training complete. Best val acc: {:.2f}".format(best_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--ssm-state-dim', type=int, default=256)
    parser.add_argument('--ssm-out-dim', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-path', type=str, default='best_cifar_ssm.pth')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    # main(args)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print("Device:", device)

    # 1) Train baseline model if it doesn't exist
    if not os.path.exists(args.save_path):
        print("No baseline checkpoint found, training model...")
        main(args)   # your existing training loop

    # 2) Load baseline model
    trainloader, testloader = get_dataloaders(batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              augment=not args.no_augment)

    model = CIFAR_SSM_Classifier(ssm_state_dim=args.ssm_state_dim,
                                 ssm_out_dim=args.ssm_out_dim).to(device)
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded baseline model from {args.save_path}, best acc = {ckpt.get('acc', 'N/A')}")

    # 3) Build initial quantized SSM from trained SimpleSSM params
    with torch.no_grad():
        A_diag_eff = -F.softplus(model.ssm.A_unconstrained.data.clone())  # effective diag
        B_q = model.ssm.B.data.clone()
        C_q = model.ssm.C.data.clone()
        D_q = model.ssm.D.data.clone()

    quantized_ssm = None
    quant_ckpt_path = "quantized_simple_ssm_cifar.pth"

    if not os.path.exists(quant_ckpt_path):
        print("Training quantized/pruned SSM via distillation...")

        # multiple pruning stages like in MNIST code
        for j in range(4):
            print(f"\n=== Quantized SSM stage {j} ===")

            A_stage = A_diag_eff.clone()
            B_stage = B_q.clone()
            C_stage = C_q.clone()
            D_stage = D_q.clone()

            if j != 0:
                sparsity = 0.2 + j * 0.1
                print(f"Applying sparsity {sparsity:.2f}")
                A_stage, A_mask = prune_tensor(A_stage, amount=sparsity)
                B_stage, B_mask = prune_tensor(B_stage, amount=sparsity)
                C_stage, C_mask = prune_tensor(C_stage, amount=sparsity)
                D_stage, D_mask = prune_tensor(D_stage, amount=sparsity)
            else:
                A_mask = torch.ones_like(A_stage)
                B_mask = torch.ones_like(B_stage)
                C_mask = torch.ones_like(C_stage)
                D_mask = torch.ones_like(D_stage)

            child = QuantizedSimpleSSM(A_stage, B_stage, C_stage, D_stage).to(device)
            optimizer = torch.optim.Adam(child.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            num_epochs = 20
            for epoch in range(num_epochs):
                child.train()
                model.eval()
                pbar = tqdm(trainloader, desc=f"QSSM Stage {j} Epoch {epoch+1}/{num_epochs}", leave=False)
                for images, _ in pbar:
                    images = images.to(device)

                    # stem + pooling to produce sequence input (B, 128, 32)
                    with torch.no_grad():
                        z = model.stem(images)           # (B,128,32,32)
                        z = model.pool_width_to_1(z)     # (B,128,32,1)
                        z = z.squeeze(-1)                # (B,128,32)
                    Bz, input_dim, L = z.shape

                    # teacher & student states
                    state_teacher = torch.zeros(Bz, model.ssm.state_dim, device=device)
                    state_child = torch.zeros(Bz, child.A_diag.numel(), device=device)

                    loss = 0.0
                    for t in range(L):
                        u_t = z[:, :, t]   # (B,input_dim)

                        # teacher single step
                        state_teacher, y_teacher = model.ssm.single_step(state_teacher, u_t)

                        # student hidden update + output
                        state_child, _ = child.hidden_update(state_child, u_t)
                        y_child = child.get_output(state_child, u_t)

                        loss = loss + criterion(state_child, state_teacher.detach()) \
                                   + criterion(y_child, y_teacher.detach())

                    loss = loss / L
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # re-apply masks to enforce sparsity
                    with torch.no_grad():
                        child.A_diag.mul_(A_mask)
                        child.B.mul_(B_mask)
                        child.C.mul_(C_mask)
                        child.D.mul_(D_mask)

                    pbar.set_postfix(loss=loss.item())

            # checkpoint this stage's child; keep last child as quantized_ssm
            quantized_ssm = child

            # current sparsity
            total_params = child.A_diag.numel() + child.B.numel() + child.C.numel() + child.D.numel()
            zero_params = (child.A_diag == 0).sum().item() + \
                          (child.B == 0).sum().item() + \
                          (child.C == 0).sum().item() + \
                          (child.D == 0).sum().item()
            sparsity_final = zero_params / total_params
            print(f"Quantized SimpleSSM sparsity after stage {j}: {sparsity_final*100:.2f}%")

            # quick eval of quantized SSM integrated into classifier
            child.eval()
            correct, total = 0, 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc=f"Testing Quantized SSM Stage {j}", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    # feeder: stem -> our quantized SSM -> head
                    z = model.stem(images)           # (B,128,32,32)
                    z = model.pool_width_to_1(z)     # (B,128,32,1)
                    z = z.squeeze(-1)                # (B,128,32)
                    Bz, input_dim, L = z.shape

                    state_q = torch.zeros(Bz, child.A_diag.numel(), device=device)
                    outputs = []
                    for t in range(L):
                        u_t = z[:, :, t]
                        state_q, _ = child.hidden_update(state_q, u_t)
                        y_t = child.get_output(state_q, u_t)
                        outputs.append(y_t)

                    y_seq = torch.stack(outputs, dim=2)     # (B, out_dim, L)
                    logits = model.head(y_seq)              # reuse existing head
                    preds = logits.argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100*correct/total)

            test_acc = correct / total
            print(f"Quantized SSM Stage {j} Test Acc = {test_acc:.4f}")

        # save last child as final quantized SSM
        torch.save(quantized_ssm.state_dict(), quant_ckpt_path)
        print("Saved quantized SSM to", quant_ckpt_path)

    else:
        print("Quantized SSM model already exists at", quant_ckpt_path)
        with torch.no_grad():
            A_diag_eff = -F.softplus(model.ssm.A_unconstrained.data.clone())
            B_q = model.ssm.B.data.clone()
            C_q = model.ssm.C.data.clone()
            D_q = model.ssm.D.data.clone()
        quantized_ssm = QuantizedSimpleSSM(A_diag_eff, B_q, C_q, D_q).to(device)
        quantized_ssm.load_state_dict(torch.load(quant_ckpt_path, map_location=device))
        quantized_ssm.eval()

    # ------------------------------------------------------------
    # PRINT SPARSITY OF QUANTIZED SSM
    # ------------------------------------------------------------
    with torch.no_grad():
        A_nonzero = (quantized_ssm.A_diag != 0).sum().item()
        B_nonzero = (quantized_ssm.B != 0).sum().item()
        C_nonzero = (quantized_ssm.C != 0).sum().item()
        D_nonzero = (quantized_ssm.D != 0).sum().item()

        A_total = quantized_ssm.A_diag.numel()
        B_total = quantized_ssm.B.numel()
        C_total = quantized_ssm.C.numel()
        D_total = quantized_ssm.D.numel()

        total_nonzero = A_nonzero + B_nonzero + C_nonzero + D_nonzero
        total_params = A_total + B_total + C_total + D_total

        print("\n========== Quantized SSM Sparsity ==========")
        print(f"A_diag sparsity: {(1 - A_nonzero / A_total) * 100:.2f}%")
        print(f"B sparsity:      {(1 - B_nonzero / B_total) * 100:.2f}%")
        print(f"C sparsity:      {(1 - C_nonzero / C_total) * 100:.2f}%")
        print(f"D sparsity:      {(1 - D_nonzero / D_total) * 100:.2f}%")
        print("--------------------------------------------")
        print(f"Total sparsity:  {(1 - total_nonzero / total_params) * 100:.2f}%")
        print("============================================\n")


    # ------------------------------------------------------------
    # TEST ACCURACY OF QUANTIZED SSM IN THE FULL CLASSIFIER
    # ------------------------------------------------------------
    correct = 0
    total = 0
    quantized_ssm.eval()
    model.eval()

    grad_collector = Correction_Module_dense(k=4)
    external_corrector = Correction_Module_ssm()
    diff_vals = []

    with torch.no_grad():
        pbar = tqdm(testloader, desc="Eval Quantized SSM", leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            if images.size(0) != args.batch_size:
                continue

            # Run stem -> quantized SSM -> head
            z = model.stem(images)               # (B,128,32,32)
            z = model.pool_width_to_1(z)         # (B,128,32,1)
            z = z.squeeze(-1)                    # (B,128,32)

            Bz, input_dim, L = z.shape
            teacher_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
            state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

            outputs = []
            model_outputs = []
            for t in range(L):
                u_t = z[:, :, t]                 # (B, input_dim)

                teacher_state, _ = model.ssm.hidden_update(teacher_state, u_t)
                state, _ = quantized_ssm.hidden_update(state, u_t)

                grad_collector.compute_grad(teacher_state, f"ssm_state_{t}")
                diff_vals.append(torch.abs(teacher_state - state))

                y_t = quantized_ssm.get_output(state, u_t)
                outputs.append(y_t)

                model_y_t = model.ssm.get_output(teacher_state, u_t)
                model_outputs.append(model_y_t)

            y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
            external_corrector.compute_grad(y_seq, layer_name="ssm_output")

            logits = model.head(y_seq)
            preds = logits.argmax(dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)
            pbar.set_postfix(acc=100 * correct / total)

    test_acc = 100 * correct / total
    print(f"Quantized SSM Model Test Accuracy = {test_acc:.2f}%")

    diff_vals = torch.stack(diff_vals, dim=0)  # shape (#steps, B, state_dim)

    mu_diff = diff_vals.mean().item()
    std_diff = diff_vals.std().item()

    print(f"\n========== Δ Statistics Between Teacher and Quantized SSM ==========")
    print(f"Mean(|Δ|)     = {mu_diff:.6f}")
    print(f"Std(|Δ|)      = {std_diff:.6f}")
    print("==========================================================\n")

    k = 5

    data = {}
    num_tests = 100

    data["nominal"] = {}
    data["nominal"]["acc"] = []
    data["nominal"]["time"] = []
    for test_num in range(num_tests):
        exec_times = []

        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
            for images, targets in pbar:
                images = images.to(device)
                targets = targets.to(device)

                if images.size(0) != args.batch_size:
                    continue

                # Run stem -> quantized SSM -> head
                z = model.stem(images)               # (B,128,32,32)
                z = model.pool_width_to_1(z)         # (B,128,32,1)
                z = z.squeeze(-1)                    # (B,128,32)

                Bz, input_dim, L = z.shape
                model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                outputs = []
                for t in range(L):
                    start = time.perf_counter()

                    u_t = z[:, :, t]                 # (B, input_dim)

                    model_state, mask = model.ssm.hidden_update(model_state, u_t)
                    # quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=True, error_rate=1e-3)

                    model_state = nan_checker(model_state)
                    # quant_state = nan_checker(quant_state)
                    # quant_state[quant_mask] = 0

                    # model_state = grad_collector(model_state, f"ssm_state_{t}")
                    # diff = torch.abs(model_state - quant_state)
                    # mask = torch.abs(diff - mu_diff) > k * std_diff
                    # model_state[mask] = quant_state[mask]

                    y_t = model.ssm.get_output(model_state, u_t)
                    outputs.append(y_t)

                    end = time.perf_counter()
                    exec_time = end - start
                    exec_times.append(exec_time)

                y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                # y_seq = external_corrector(y_seq, "ssm_output")

                logits = model.head(y_seq)
                preds = logits.argmax(dim=1)

                correct += (preds == targets).sum().item()
                total += targets.size(0)
                pbar.set_postfix(acc=100 * correct / total)

        test_acc = correct / total
        median_exec_time = stats.median(exec_times) * 1000

        data["nominal"]["acc"].append(test_acc)
        data["nominal"]["time"].append(median_exec_time)
    
    error_rates = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

    for error_rate in error_rates:
        key = f"no-corr-{error_rate}-output"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, out_inject=True, error_rate=error_rate)
                        # quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=True, error_rate=1e-3)

                        model_state = nan_checker(model_state)
                        # quant_state = nan_checker(quant_state)
                        # quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"no-corr-{error_rate}-weight"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, weight_inject=True, error_rate=error_rate)
                        # quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=True, error_rate=1e-3)

                        model_state = nan_checker(model_state)
                        # quant_state = nan_checker(quant_state)
                        # quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"external-zero-corr-{error_rate}-output"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, out_inject=True, error_rate=error_rate)
                        # quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=True, error_rate=1e-3)

                        model_state = nan_checker(model_state)
                        # quant_state = nan_checker(quant_state)
                        # quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"external-zero-corr-{error_rate}-weight"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, weight_inject=True, error_rate=error_rate)
                        # quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=True, error_rate=1e-3)

                        model_state = nan_checker(model_state)
                        # quant_state = nan_checker(quant_state)
                        # quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"internal-zero-corr-{error_rate}-output"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, out_inject=True, error_rate=error_rate)
                        # quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=True, error_rate=1e-3)

                        model_state = nan_checker(model_state)
                        # quant_state = nan_checker(quant_state)
                        # quant_state[quant_mask] = 0

                        model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"internal-zero-corr-{error_rate}-weight"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, weight_inject=True, error_rate=error_rate)
                        # quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=True, error_rate=1e-3)

                        model_state = nan_checker(model_state)
                        # quant_state = nan_checker(quant_state)
                        # quant_state[quant_mask] = 0

                        model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-clean-corr-{error_rate}-output"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, out_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=False, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        diff = torch.abs(model_state - quant_state)
                        mask = torch.abs(diff - mu_diff) > k * std_diff
                        model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-clean-corr-{error_rate}-weight"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, weight_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, weight_inject=False, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        diff = torch.abs(model_state - quant_state)
                        mask = torch.abs(diff - mu_diff) > k * std_diff
                        model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-clean-checksum-corr-{error_rate}-output"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, out_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=False, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-clean-checksum-corr-{error_rate}-weight"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, weight_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, weight_inject=False, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-clean-gradient-corr-{error_rate}-output"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, out_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=False, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        model_state = grad_collector(model_state, f"ssm_state_{t}", replace_tensor=quant_state)
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-clean-gradient-corr-{error_rate}-weight"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, weight_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, weight_inject=False, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        model_state = grad_collector(model_state, f"ssm_state_{t}", replace_tensor=quant_state)
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-dirty-corr-{error_rate}-output"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, out_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=False, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        diff = torch.abs(model_state - quant_state)
                        mask = torch.abs(diff - mu_diff) > k * std_diff
                        model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-dirty-corr-{error_rate}-weight"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, weight_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, weight_inject=True, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        diff = torch.abs(model_state - quant_state)
                        mask = torch.abs(diff - mu_diff) > k * std_diff
                        model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-dirty-checksum-corr-{error_rate}-output"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, out_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=True, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-dirty-checksum-corr-{error_rate}-weight"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, weight_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, weight_inject=True, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        # model_state = grad_collector(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)
        
        key = f"backup-dirty-gradient-corr-{error_rate}-output"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, out_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, out_inject=True, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        model_state = grad_collector(model_state, f"ssm_state_{t}", replace_tensor=quant_state)
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)

        key = f"backup-dirty-gradient-corr-{error_rate}-weight"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []

        for test_num in range(num_tests):
            exec_times = []

            correct = 0
            total = 0
            with torch.no_grad():
                pbar = tqdm(testloader, desc="Eval Corrected SSM", leave=False)
                for images, targets in pbar:
                    images = images.to(device)
                    targets = targets.to(device)

                    if images.size(0) != args.batch_size:
                        continue

                    # Run stem -> quantized SSM -> head
                    z = model.stem(images)               # (B,128,32,32)
                    z = model.pool_width_to_1(z)         # (B,128,32,1)
                    z = z.squeeze(-1)                    # (B,128,32)

                    Bz, input_dim, L = z.shape
                    model_state = model.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    quant_state = torch.zeros(Bz, quantized_ssm.A_diag.numel(), device=device)

                    outputs = []
                    for t in range(L):
                        start = time.perf_counter()

                        u_t = z[:, :, t]                 # (B, input_dim)

                        model_state, mask = model.ssm.hidden_update(model_state, u_t, weight_inject=True, error_rate=error_rate)
                        quant_state, quant_mask = quantized_ssm.hidden_update(quant_state, u_t, weight_inject=True, error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        quant_state = nan_checker(quant_state)
                        quant_state[quant_mask] = 0

                        model_state = grad_collector(model_state, f"ssm_state_{t}", replace_tensor=quant_state)
                        # diff = torch.abs(model_state - quant_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quant_state[mask]

                        y_t = model.ssm.get_output(model_state, u_t)
                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_time = end - start
                        exec_times.append(exec_time)

                    y_seq = torch.stack(outputs, dim=2)  # (B, output_dim, L)
                    # y_seq = external_corrector(y_seq, "ssm_output")

                    logits = model.head(y_seq)
                    preds = logits.argmax(dim=1)

                    correct += (preds == targets).sum().item()
                    total += targets.size(0)
                    pbar.set_postfix(acc=100 * correct / total)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)
    
    output_path = "stats_out.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)