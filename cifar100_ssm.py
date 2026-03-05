import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import math
import wandb
import json

def nan_checker(x):
    nan_check = torch.isnan(x)
    inf_check = torch.isinf(x)
    if torch.sum(nan_check) or torch.sum(inf_check):
        x = x.masked_fill_(nan_check,0)
        x = x.masked_fill_(inf_check,0)
    return x  

def flip_bits(A, error_rate=1e-4, clamp_val=1e6):
    orig_dtype = A.dtype
    device = A.device

    # Ensure contiguous float32
    A_fp32 = A.clone().float().contiguous()

    # Flatten
    flat = A_fp32.view(-1)

    # Reinterpret bits
    flat_int = flat.view(torch.int32)

    # Decide flips
    flip_mask = torch.rand(flat_int.shape[0], device=device) < error_rate
    if not flip_mask.any():
        return A

    # Choose random bit per flipped element
    bit_positions = torch.randint(
        0, 32, (flip_mask.sum(),),
        device=device, dtype=torch.int32
    )

    # Apply XOR only where needed
    flat_int[flip_mask] ^= (1 << bit_positions)

    # Restore shape
    out = flat_int.view(torch.float32).view_as(A_fp32)

    # Sanitize
    out = torch.nan_to_num(out, nan=0.0, posinf=clamp_val, neginf=-clamp_val)

    return out.to(orig_dtype).to(device)

# ---------- SSM Layer ----------
class SSM(nn.Module):
    def __init__(self, input_dim, state_dim, output_dim, init_scale=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim

        self.A_diag = nn.Parameter(torch.randn(state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * init_scale / math.sqrt(input_dim))
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * init_scale / math.sqrt(input_dim))
        self.D = nn.Parameter(torch.zeros(output_dim, input_dim))

        self.h0 = nn.Parameter(torch.zeros(state_dim))

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, u):
        Bz, input_dim, L = u.shape
        device = u.device

        state = self.h0.unsqueeze(0).expand(Bz, -1).contiguous()
        u_t = u.permute(1, 0, 2)  # (L, B, input_dim)

        A_diag = -F.softplus(self.A_diag)
        A = torch.diag(A_diag)

        outputs = []
        for t in range(L):
            ut = u_t[t]  # (B, input_dim)

            state = F.linear(state, A) + F.linear(ut, self.B)
            y_t = F.linear(state, self.C) + F.linear(ut, self.D)

            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        activation = self.activation(y)
        norm = self.norm(activation)
        return torch.nan_to_num(norm, nan=0.0, posinf=1e6, neginf=-1e6)

    def single_step(self, state, u, inject_type="none", error_rate=0):
        if state is None:
           state = self.h0.unsqueeze(0).expand(B, -1).contiguous()
        x_next = self.hidden_update(state, u, inject_type=inject_type, error_rate=error_rate)
        y = self.get_output(x_next, u)
        return x_next, y
    
    def hidden_update(self, x, u, inject_type="none", error_rate=0):
        B = self.B
        A_diag = self.A_diag

        if inject_type == "weight":
            A_diag = flip_bits(A_diag.detach(), error_rate=error_rate)
            B = flip_bits(B.detach(), error_rate=error_rate)

        A_diag = -F.softplus(A_diag)
        A = torch.diag(A_diag)

        left_term = F.linear(x, A)
        right_term = F.linear(u, B)
        
        if inject_type == "output":
            left_term = flip_bits(left_term.detach(), error_rate=error_rate)
            right_term = flip_bits(right_term.detach(), error_rate=error_rate)

        x_next = left_term + right_term
        return x_next
    
    def get_output(self, x, u, inject_type="none", error_rate=1e-3):
        C, D = self.C, self.D

        if inject_type == "weight":
            C = flip_bits(C.detach(), error_rate=error_rate)
            D = flip_bits(D.detach(), error_rate=error_rate)
        
        left_term = F.linear(x, C)
        right_term = F.linear(u, D)

        if inject_type == "output":
            left_term = flip_bits(left_term.detach(), error_rate=error_rate)
            right_term = flip_bits(right_term.detach(), error_rate=error_rate)

        output = left_term + right_term
        return output

# ---------- Classifier Model ----------
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SSMClassifier(nn.Module):
    def __init__(self, input_dim=128, ssm_state_dim=256, ssm_out_dim=256, n_classes=100):
        super().__init__()
        self.ssm = SSM(input_dim=input_dim, state_dim=ssm_state_dim, output_dim=ssm_out_dim)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # average over sequence dimension -> (B, ssm_out_dim, 1)
            nn.Flatten(),
            nn.Linear(ssm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        y = self.ssm(x)
        y = y.transpose(1, 2).contiguous()
        return self.fc(y)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, ssm_state_dim=512, ssm_out_dim=256, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.classifier = SSMClassifier(
            input_dim=64,
            ssm_state_dim=ssm_state_dim,
            ssm_out_dim=ssm_out_dim,
            n_classes=num_classes
        )

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _process_to_ssm(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        B, C, H, W = out.shape  # (B, 64, 8, 8)
        out = out.view(B, C, H * W).permute(0, 2, 1)    # flatten spatial output to sequential
        return out

    def forward(self, x):
        out = self._process_to_ssm(x)
        out = self.classifier(out)
        return out

def resnet56(num_classes=100):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)

def ssm_checksum_verify(x_new, x_old, A, u, B):
    A_s = A.sum(dim=0)
    B_s = B.sum(dim=0)

    # print(A_s, B_s)
    # print(x_old, u)

    expected_sum = x_old @ A_s.t() + u @ B_s.t()
    actual_sum = x_new.sum(dim=1)

    return torch.norm(actual_sum - expected_sum, dim=-1, p=2)

def quantize_to_int8(x, scale):
    q = torch.round(x * scale)
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8)

def dequantize_from_int8(q, scale):
    return q.float() / scale

def ste_mask(scores, threshold):
    hard = (scores >= threshold).float()
    return hard + (scores - scores.detach()) - (threshold - threshold.detach())

class WeightAdapter(nn.Module):
    def __init__(self, out_features):
       super().__init__()
       self.scale = nn.Parameter(torch.ones(out_features, 1))
       self._threshold = nn.Parameter(torch.tensor(0.001))
       
    @property
    def threshold(self):
        return F.softplus(self._threshold, beta=15, threshold=0.01)

    def forward(self, weight, use_mask=False):
        w_adapted = weight * self.scale
        scores = torch.abs(w_adapted).mean(dim=1, keepdim=True)
        
        if use_mask:
            th = self.threshold
            mask = ste_mask(scores, th)
            w_final = w_adapted * mask
        else:
            w_final = w_adapted

        return w_final

class ParamWithAdapter(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = nn.Parameter(param)
        self.adapter = WeightAdapter(param.shape[0])

    def get_weight(self, use_mask=False):
        w_final = self.adapter(self.param.data, use_mask)
        return w_final

    def forward(self, x, use_mask=False):
        w_final = self.get_weight(use_mask=use_mask)
        out = F.linear(x, w_final)
        return out

class ParamWithAdapterDiagonal(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = nn.Parameter(param)
        self.adapter = WeightAdapter(param.shape[0])

    def get_weight(self, use_mask=False):
        w = -F.softplus(self.param.data)
        w = torch.diag(w)
        w_final = self.adapter(w, use_mask)
        return w_final

    def forward(self, x, use_mask=False):
        w_final = self.get_weight(use_mask=use_mask)
        out = F.linear(x, w_final)
        return out

class StateSpaceWithAdapter(nn.Module):
    def __init__(self, A_diag, B, C, D):
        super().__init__()
        self.A_diag = ParamWithAdapterDiagonal(A_diag)
        self.B = ParamWithAdapter(B)
        self.C = ParamWithAdapter(C)
        self.D = ParamWithAdapter(D)

    def forward(self, x, u, use_mask=False):
        x_next = self.hidden_update(x, u, use_mask=use_mask)
        y = self.get_output(x_next, u, use_mask=use_mask)
        return x_next, y
    
    def hidden_update(self, x, u, use_mask=False):
        left_term = self.A_diag(x, use_mask=use_mask)
        right_term = self.B(u, use_mask=use_mask)
        x_next = left_term + right_term
        return x_next

    def get_output(self, x, u, use_mask=False):
        left_term = self.C(x, use_mask=use_mask)
        right_term = self.D(u, use_mask=use_mask)
        y = left_term + right_term
        return y

def quantize_tensor(X, N, eps=1e-8):
    max_val = torch.max(torch.abs(X))
    denom = (2 ** (N - 1) - 1)

    s = max_val / denom
    s_safe = torch.where(s == 0, torch.ones_like(s), s)

    X_bar = torch.clamp(
        torch.round(X / s_safe),
        -2 ** (N - 1),
        2 ** (N - 1) - 1
    )

    # If max_val was zero → force quantized tensor to zero
    X_bar = torch.where(max_val == 0, torch.zeros_like(X_bar), X_bar)

    return X_bar, s_safe

class QuantizedSSM(nn.Module):
    def __init__(self, A, B, C, D, N=4):
        super().__init__()

        self.N = N

        A_bar, s_A = quantize_tensor(A, N)
        B_bar, s_B = quantize_tensor(B, N)
        C_bar, s_C = quantize_tensor(C, N)
        D_bar, s_D = quantize_tensor(D, N)

        self.A_diag = nn.Parameter(A_bar)
        self.B = nn.Parameter(B_bar)
        self.C = nn.Parameter(C_bar)
        self.D = nn.Parameter(D_bar)
        self.s_A = s_A
        self.s_B = s_B
        self.s_C = s_C
        self.s_D = s_D

        self.tau = float('inf')
        self.checksum_norms = []
        self.active = False

        self.x_min = float('-inf')
        self.x_max = float('inf')

    def forward(self, x, u):
        x_next = self.hidden_update(x, u)
        y = self.get_output(x_next, u)
        return x_next, y
    
    def hidden_update(self, x, u, inject_type="none", error_rate=0):
        A_q_diag = self.A_diag
        B_q = self.B

        # memory access errors
        if inject_type == "weight":
            A_q_diag = torch.clamp(
                flip_bits(A_q_diag.detach(), error_rate=error_rate),
                -2**(self.N-1), 2**(self.N-1)-1
            )
            B_q = torch.clamp(
                flip_bits(B_q.detach(), error_rate=error_rate),
                -2**(self.N-1), 2**(self.N-1)-1
            )

        # dequantize weights
        A_diag = A_q_diag * self.s_A
        B = B_q * self.s_B

        # diagonalize A
        A_diag = -F.softplus(A_diag)
        A = torch.diag(A_diag)

        # compute
        left_term = F.linear(x, A)
        right_term = F.linear(u, B)

        # computation output errors
        if inject_type == "output":
            left_term = flip_bits(left_term, error_rate=error_rate)
            right_term = flip_bits(right_term, error_rate=error_rate)

        # get hidden state 
        x_next = left_term + right_term

        if not self.training:
            v_norms = ssm_checksum_verify(x_next, x, A, u, B)
            if not self.active:
                self.checksum_norms.append(v_norms.cpu().tolist())
        
        if self.active:
            exceed_mask = v_norms > self.tau
            if exceed_mask.any():
                x_next[exceed_mask] = self.RGD(x_next[exceed_mask])

        return x_next

    def get_output(self, x, u):
        # dequantize weights
        C = self.C * self.s_C
        D = self.D * self.s_D

        # get output
        left_term = F.linear(x, C)
        right_term = F.linear(u, D)

        output = left_term + right_term
        return output
    
    def RGD(self, x):
        mask = (x >= self.x_min) & (x <= self.x_max)
        return x * mask
    
    def clamp(self, x):
        return torch.clamp(x, min=self.x_min, max=self.x_max)
    
    def set_bounds(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
    
    def compute_tau(self):
        self.active = True
        if self.checksum_norms:
            norms = torch.tensor(self.checksum_norms)
            self.tau = norms.mean().item() + 4*norms.std().item()
        else:
            self.tau = 0.0
        return self.tau

    def get_sparsity(self):
        total_zeros = 0
        total_params = 0
        
        for layer in [self.B, self.C, self.D]:
            w = layer.data
            total_zeros += (w < 1e-8).sum().item()
            total_params += w.numel()
        
        return total_zeros / total_params

class RunningStats:
    def __init__(self, k=5):
        self.mean = None
        self.M2 = None
        self.count = 0
        self.k = k

    def update(self, x):
        batch_count = x.shape[0]
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)

        if self.mean is None:
            self.mean = batch_mean
            self.M2 = batch_var * batch_count
            self.count = batch_count
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count

            self.mean += delta * batch_count / total
            self.M2 += (
                batch_var * batch_count
                + delta**2 * self.count * batch_count / total
            )
            self.count = total

    def cantelli(self, x, two_sided=True):
        if self.mean is None:
            raise ValueError()
        
        upper = self.mean + self.k * self.std

        if two_sided:
            lower = self.mean - self.k * self.std
            return (x > upper) | (x < lower)
        else:
            return x > upper

    @property
    def variance(self):
        return self.M2 / max(self.count, 1)

    @property
    def std(self):
        return torch.sqrt(self.variance.clamp_min(1e-8))

# ---------------------------
# Training / utilities
# ---------------------------
def get_dataloaders(batch_size=128, num_workers=4, augment=True):
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_transforms = []
    if augment:
        train_transforms += [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip()
        ]
    train_transforms += [T.ToTensor(), T.Normalize(mean, std)]
    train_transform = T.Compose(train_transforms)
    test_transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def train_model(args):
    device = args.device

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    model = resnet56(num_classes=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_acc = 0.0

    # ------------------------
    # Resume from checkpoint
    # ------------------------
    if args.resume is not None:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]

        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    # ------------------------
    # Eval-only mode
    # ------------------------
    if args.eval_only:
        if args.resume is None:
            raise ValueError("Must provide --resume checkpoint for evaluation.")
        return model, test_loader, train_loader

    # ------------------------
    # Training loop
    # ------------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # ------------------------
        # Evaluation
        # ------------------------
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Test]", leave=False):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        test_acc = correct / total

        scheduler.step()

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, "
            f"Train Acc={train_acc:.4f}, "
            f"Test Acc={test_acc:.4f}"
        )

        # ------------------------
        # Save best checkpoint
        # ------------------------
        if test_acc > best_acc:
            best_acc = test_acc
            save_path = args.checkpoint

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_acc": best_acc,
            }, save_path)

            print(f"Saved best model to {save_path}")

    return model, test_loader, train_loader

def evaluate_child(model, child, test_loader, use_mask=False):
    model.eval()
    child.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing Child SSM Model", leave=False):
            x, y = x.to(device), y.to(device)
            u = model._process_to_ssm(x)

            Bz, L, input_dim = u.shape
            u_t = u.permute(1, 0, 2)
            
            state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
            outputs = []
            
            for t in range(L):
                ut = u_t[t]
                if isinstance(child, StateSpaceWithAdapter):
                    state, y_t = child(state, ut, use_mask=use_mask)
                else:
                    state, y_t = child(state, ut)
                outputs.append(y_t)
                
            output = torch.stack(outputs, dim=1)
            # output = output.to(torch.float32)
            output = model.classifier.ssm.activation(output)
            output = model.classifier.ssm.norm(output)
            output = output.transpose(1, 2).contiguous()
            logits = model.classifier.fc(output)
            preds = logits.argmax(dim=1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
            
    test_acc = correct / total
    return test_acc

import time
import statistics as stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="best_cifar100_ssm.pth")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()
    model, test_loader, train_loader = train_model(args)

    device = args.device

    A_diag = model.classifier.ssm.A_diag.data.clone()
    # print(A.shape, A)
    B = model.classifier.ssm.B.data.clone()
    C = model.classifier.ssm.C.data.clone()
    D = model.classifier.ssm.D.data.clone()

    desired_quantized_model = f"quantized_ssm_structured_cifar100.pth"
    child = StateSpaceWithAdapter(A_diag, B, C, D).to(device)

    params = {"A": child.A_diag.param, "B": child.B.param, "C": child.C.param, "D": child.D.param}

    base_acc = evaluate_child(model, child, test_loader)
    acc_threshold = base_acc - 0.1

    print(f"Initial Accuracy: {base_acc*100:.2f}%")

    lr = 3e-5
    num_epochs = 10
    lambda_reg = 1e-6

    for name, param in child.named_parameters():
        param.requires_grad = ("threshold" not in name and "quant" not in name)

    optimizer = torch.optim.Adam(
        [p for p in child.parameters() if p.requires_grad],
        lr=lr
    )
    criterion = nn.MSELoss()

    # wandb.init(
    #     project="quantized-ssm-cifar100",
    #     config={
    #         "epochs": num_epochs,
    #         "lambda_reg": lambda_reg,
    #     }
    # )

    # wandb.define_metric("train_step")
    # wandb.define_metric("prune_step")
    # wandb.define_metric("epoch")
    # wandb.define_metric("prune_epoch")

    # wandb.define_metric("train_step/*", step_metric="train_step")
    # wandb.define_metric("train/*", step_metric="epoch")
    # wandb.define_metric("prune/*", step_metric="prune_epoch")
    # wandb.define_metric("prune_step/*", step_metric="prune_step")

    trained_unpruned_child_model = f"trained_{desired_quantized_model}"
    if not os.path.exists(trained_unpruned_child_model):
        plt.figure()
        for name, param in params.items():
            values = param.detach().cpu().numpy().ravel()
            plt.hist(values, bins=50, alpha=0.5, label=name)
        plt.legend()
        plt.xlabel("Weight value")
        plt.ylabel("Frequency")
        plt.title(f"Weight Distribution (Before Regularization)")
        plt.show()

        global_step = 0
        do_prune = True
        for epoch in range(num_epochs):
            child.train()
            model.eval()
            for x, y in tqdm(train_loader, desc=f"Child SSM Training Epoch {epoch+1}/{num_epochs}", leave=False):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                u = model._process_to_ssm(x)

                Bz, L, input_dim = u.shape
                u_t = u.permute(1, 0, 2)
                
                state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                parent_state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                outputs = []

                distill_loss = 0.0

                for t in range(L):
                    ut = u_t[t]
                    state, y_t = child(state, ut)
                    with torch.no_grad():
                        parent_state, parent_y_t = model.classifier.ssm.single_step(parent_state, ut)

                    distill_loss += criterion(state, parent_state.detach()) + criterion(y_t, parent_y_t.detach())

                distill_loss /= L
                l1_A = child.A_diag.param.abs().sum()
                l1_B = child.B.param.abs().sum()
                l1_C = child.C.param.abs().sum()
                l1_D = child.D.param.abs().sum()
                l1_reg = l1_A + l1_B + l1_C + l1_D

                if do_prune:
                    loss = distill_loss + lambda_reg * l1_reg
                else:
                    loss = distill_loss
                
                loss.backward()
                optimizer.step()

                # wandb.log({
                #     "train_step": global_step,
                #     "train_step/loss": loss.item(),
                #     "train_step/distill_loss": distill_loss.item(),
                #     "train_step/l1_total": l1_reg.item(),
                #     "train_step/l1_A": l1_A.item(),
                #     "train_step/l1_B": l1_B.item(),
                #     "train_step/l1_C": l1_C.item(),
                #     "train_step/l1_D": l1_D.item(),
                # })

                global_step += 1

            test_acc = evaluate_child(model, child, test_loader)
            print(f"Child SSM Model Test Acc at Epoch {epoch+1}: {test_acc*100:.2f}%")

            if do_prune and test_acc < acc_threshold:
                do_prune = False
                print("switching to train only")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-3
            elif not do_prune and test_acc > acc_threshold:
                do_prune = True
                print("resuming regularization")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5

            current_lr = optimizer.param_groups[0]['lr']
            # wandb.log({
            #     "epoch": epoch,
            #     "train/train_test_accuracy": test_acc,
            #     "train/lr": current_lr,
            # })

        plt.figure()
        for name, param in params.items():
            values = param.detach().cpu().numpy().ravel()
            plt.hist(values, bins=50, alpha=0.5, label=name)
        plt.legend()
        plt.xlabel("Weight value")
        plt.ylabel("Frequency")
        plt.title(f"Weight Distribution (After Regularization)")
        plt.show()

        torch.save(child.state_dict(), trained_unpruned_child_model)
    else:
        child.load_state_dict(torch.load(trained_unpruned_child_model))

    pruned_child_model = f"pruned_{desired_quantized_model}"
    if not os.path.exists(pruned_child_model):
        num_epochs_prune = 20
        lr_threshold = 1e-5
        lambda_sparse = 1e-6

        for name, param in child.named_parameters():
            param.requires_grad = ("threshold" in name)

        prune_optim = torch.optim.Adam(
            [p for p in child.parameters() if p.requires_grad],
            lr=lr_threshold
        )

        base_acc = evaluate_child(model, child, test_loader)
        target_acc = base_acc - 0.05
        print(f"Baseline Accuracy: {base_acc*100:.2f}%")
        print(f"Target Minimum Accuracy: {target_acc*100:.2f}%")

        # print(child.A.adapter._threshold)

        global_step = 0
        for epoch in range(num_epochs_prune):
            child.train()
            model.eval()
            for x, y in tqdm(train_loader, desc=f"Child SSM Pruning Epoch {epoch+1}/{num_epochs_prune}", leave=False):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                u = model._process_to_ssm(x)

                Bz, L, input_dim = u.shape
                u_t = u.permute(1, 0, 2)
                
                state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                parent_state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                outputs = []

                distill_loss = 0.0

                for t in range(L):
                    ut = u_t[t]
                    state, y_t = child(state, ut)
                    with torch.no_grad():
                        parent_state, parent_y_t = model.classifier.ssm.single_step(parent_state, ut)

                    distill_loss += criterion(state, parent_state.detach()) + criterion(y_t, parent_y_t.detach())
        
                distill_loss /= L
                
                sparsity_push = -lambda_sparse * (
                    child.B.adapter._threshold +
                    child.C.adapter._threshold +
                    child.D.adapter._threshold
                )

                loss = distill_loss + sparsity_push

                # wandb.log({
                #     "prune_step": global_step,
                #     "prune_step/loss": loss.item(),
                # })
                global_step += 1

                prune_optim.zero_grad()
                loss.backward()
                prune_optim.step()

            current_acc = evaluate_child(model, child, test_loader, use_mask=True)
            thresholds = [
                child.B.adapter.threshold.item(),
                child.C.adapter.threshold.item(),
                child.D.adapter.threshold.item()
            ]

            print(f"Epoch {epoch+1:02d} | "
                f"Masked Acc: {current_acc*100:.2f}% | "
                f"Thresholds: {[round(t,4) for t in thresholds]}")

            # Prevent over-pruning
            if current_acc < target_acc:
                lambda_sparse = 1e-8
                print("⚠ Accuracy dropped too much — soft rollback")
                with torch.no_grad():
                    child.B.adapter._threshold -= 0.005
                    child.C.adapter._threshold -= 0.005
                    child.D.adapter._threshold -= 0.005

            # wandb.log({
            #     "prune_epoch": epoch,
            #     "prune/epoch_masked_accuracy": current_acc,
            #     "prune/threshold_A": thresholds[0],
            #     "prune/threshold_B": thresholds[1],
            #     "prune/threshold_C": thresholds[2]
            # })

        torch.save(child.state_dict(), pruned_child_model)
    else:
        child.load_state_dict(torch.load(pruned_child_model))

    acc = evaluate_child(model, child, test_loader, use_mask=True)
    print(f"Pre-Quantized Accuracy of Child Model: {acc*100:.2f}%")
    # PTQ
    N=2
    quantized_ssm = QuantizedSSM(
        child.A_diag.param.data,
        child.B.get_weight(use_mask=True),
        child.C.get_weight(use_mask=True),
        child.D.get_weight(use_mask=True),
        N=N
    )
    acc = evaluate_child(model, quantized_ssm, test_loader)
    print(f"Accuracy of Quantized Model: {acc*100:.2f}%")
    sparsity = quantized_ssm.get_sparsity()
    print(f"Sparsity: {sparsity*100:.2f}%")

    model.eval()
    quantized_ssm.eval()
    correct, total = 0, 0

    x_min_running = float('inf')
    x_max_running = float('-inf')

    running_stats = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Collecting Statistics", leave=False):
            x, y = x.to(device), y.to(device)
            u = model._process_to_ssm(x)

            Bz, L, input_dim = u.shape
            u_t = u.permute(1, 0, 2)
            
            if not running_stats:
                running_stats = [RunningStats() for _ in range(L)]
            
            state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
            model_state = state.clone()
            
            outputs = []
            for t in range(L):
                ut = u_t[t]
                state, y_t = quantized_ssm(state, ut)

                model_state = model.classifier.ssm.hidden_update(model_state, ut)                
                # running_stats[t].update(model_state)

                # Compute batch quantiles
                batch_flat = state.view(-1)
                batch_x_min = batch_flat.quantile(0.01).item()
                batch_x_max = batch_flat.quantile(0.99).item()

                # Update running x_min/x_max
                x_min_running = min(x_min_running, batch_x_min)
                x_max_running = max(x_max_running, batch_x_max)

                model_output = model.classifier.ssm.get_output(model_state, ut)
                running_stats[t].update(model_output)

                outputs.append(y_t)
                
            output = torch.stack(outputs, dim=1)
            output = model.classifier.ssm.activation(output)
            output = model.classifier.ssm.norm(output)
            output = output.transpose(1, 2).contiguous()
            logits = model.classifier.fc(output)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    print(f"Quantized SSM Model Test Acc={test_acc:.4f}")

    print(f"x_min={x_min_running:.4f}, x_max={x_max_running:.4f}")
    quantized_ssm.set_bounds(x_min_running, x_max_running)

    tau = quantized_ssm.compute_tau()
    print(f"tau: {tau:.4f}")

    error_rates = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    data = {}
    NUM_TESTS = 20

    for error_rate in error_rates:
        # key = f"no-corr-{error_rate}"
        # data[key] = {}
        # data[key]["acc"] = []
        # data[key]["time"] = []
        # for test in range(NUM_TESTS):
        #     exec_times = []

        #     model.eval()
        #     quantized_ssm.eval()
        #     correct, total = 0, 0
        #     with torch.no_grad():
        #         for x, y in tqdm(test_loader, desc=f"Correction Testing {key} {test+1}/{NUM_TESTS}", leave=False):
        #             x, y = x.to(device), y.to(device)
        #             u = model._process_to_ssm(x)

        #             Bz, L, input_dim = u.shape
        #             u_t = u.permute(1, 0, 2)

        #             model_state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
        #             outputs = []

        #             for t in range(L):
        #                 start = time.perf_counter()

        #                 ut = u_t[t]
        #                 model_state = model.classifier.ssm.hidden_update(model_state, ut, inject_type="output", error_rate=error_rate)

        #                 y_t = model.classifier.ssm.get_output(model_state, ut)
        #                 outputs.append(y_t)

        #                 end = time.perf_counter()
        #                 exec_times.append(end - start)

        #             output = torch.stack(outputs, dim=1)
        #             output = model.classifier.ssm.activation(output)
        #             output = model.classifier.ssm.norm(output)
        #             output = output.transpose(1, 2).contiguous()
        #             logits = model.classifier.fc(output)
        #             preds = logits.argmax(dim=1)
        #             correct += (preds == y).sum().item()
        #             total += y.size(0)

        #     test_acc = correct / total
        #     median_exec_time = stats.median(exec_times) * 1000

        #     data[key]["acc"].append(test_acc)
        #     data[key]["time"].append(median_exec_time)
        # print(f"{key} avg acc = {sum(data[key]['acc'])/NUM_TESTS:.4f}")

        key = f"zeroing-{error_rate}"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []
        for test in range(NUM_TESTS):
            exec_times = []

            model.eval()
            quantized_ssm.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in tqdm(test_loader, desc=f"Correction Testing {key} {test+1}/{NUM_TESTS}", leave=False):
                    x, y = x.to(device), y.to(device)
                    u = model._process_to_ssm(x)

                    Bz, L, input_dim = u.shape
                    u_t = u.permute(1, 0, 2)

                    model_state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    outputs = []

                    for t in range(L):
                        start = time.perf_counter()

                        ut = u_t[t]
                        model_state = model.classifier.ssm.hidden_update(model_state, ut, inject_type="output", error_rate=error_rate)

                        # model_state = nan_checker(model_state)

                        # mask = running_stats[t].cantelli(model_state)
                        # model_state[mask] = 0

                        y_t = model.classifier.ssm.get_output(model_state, ut)

                        mask = running_stats[t].cantelli(y_t)
                        y_t[mask] = 0

                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_times.append(end - start)

                    output = torch.stack(outputs, dim=1)
                    output = model.classifier.ssm.activation(output)
                    output = model.classifier.ssm.norm(output)
                    output = output.transpose(1, 2).contiguous()
                    logits = model.classifier.fc(output)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)
        print(f"{key} avg acc = {sum(data[key]['acc'])/NUM_TESTS:.4f}")

        # key = f"backup-{error_rate}"
        # data[key] = {}
        # data[key]["acc"] = []
        # data[key]["time"] = []
        # for test in range(NUM_TESTS):
        #     exec_times = []

        #     model.eval()
        #     quantized_ssm.eval()
        #     correct, total = 0, 0
        #     with torch.no_grad():
        #         for x, y in tqdm(test_loader, desc=f"Correction Testing {key} {test+1}/{NUM_TESTS}", leave=False):
        #             x, y = x.to(device), y.to(device)
        #             u = model._process_to_ssm(x)

        #             Bz, L, input_dim = u.shape
        #             u_t = u.permute(1, 0, 2)

        #             model_state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
        #             outputs = []

        #             for t in range(L):
        #                 start = time.perf_counter()

        #                 ut = u_t[t]
        #                 quantized_state = quantized_ssm.hidden_update(model_state, ut, inject_type="output", error_rate=error_rate)
        #                 model_state = model.classifier.ssm.hidden_update(model_state, ut, inject_type="output", error_rate=error_rate)

        #                 model_state = nan_checker(model_state)
        #                 quantized_state = nan_checker(quantized_state)

        #                 mask = running_stats[t].cantelli(model_state)
        #                 model_state[mask] = quantized_state[mask]

        #                 y_t = model.classifier.ssm.get_output(model_state, ut)
        #                 outputs.append(y_t)

        #                 end = time.perf_counter()
        #                 exec_times.append(end - start)

        #             output = torch.stack(outputs, dim=1)
        #             output = model.classifier.ssm.activation(output)
        #             output = model.classifier.ssm.norm(output)
        #             output = output.transpose(1, 2).contiguous()
        #             logits = model.classifier.fc(output)
        #             preds = logits.argmax(dim=1)
        #             correct += (preds == y).sum().item()
        #             total += y.size(0)

        #     test_acc = correct / total
        #     median_exec_time = stats.median(exec_times) * 1000

        #     data[key]["acc"].append(test_acc)
        #     data[key]["time"].append(median_exec_time)
        # print(f"{key} avg acc = {sum(data[key]['acc'])/NUM_TESTS:.4f}")

    output_path = f"cifar100_stats_output_zeroing.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    error_rates = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    data = {}
    NUM_TESTS = 20

    for error_rate in error_rates:
        # key = f"no-corr-{error_rate}"
        # data[key] = {}
        # data[key]["acc"] = []
        # data[key]["time"] = []
        # for test in range(NUM_TESTS):
        #     exec_times = []

        #     model.eval()
        #     quantized_ssm.eval()
        #     correct, total = 0, 0
        #     with torch.no_grad():
        #         for x, y in tqdm(test_loader, desc=f"Correction Testing {key} {test+1}/{NUM_TESTS}", leave=False):
        #             x, y = x.to(device), y.to(device)
        #             u = model._process_to_ssm(x)

        #             Bz, L, input_dim = u.shape
        #             u_t = u.permute(1, 0, 2)

        #             model_state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
        #             outputs = []

        #             for t in range(L):
        #                 start = time.perf_counter()

        #                 ut = u_t[t]
        #                 model_state = model.classifier.ssm.hidden_update(model_state, ut, inject_type="weight", error_rate=error_rate)

        #                 y_t = model.classifier.ssm.get_output(model_state, ut)
        #                 outputs.append(y_t)

        #                 end = time.perf_counter()
        #                 exec_times.append(end - start)

        #             output = torch.stack(outputs, dim=1)
        #             output = model.classifier.ssm.activation(output)
        #             output = model.classifier.ssm.norm(output)
        #             output = output.transpose(1, 2).contiguous()
        #             logits = model.classifier.fc(output)
        #             preds = logits.argmax(dim=1)
        #             correct += (preds == y).sum().item()
        #             total += y.size(0)

        #     test_acc = correct / total
        #     median_exec_time = stats.median(exec_times) * 1000

        #     data[key]["acc"].append(test_acc)
        #     data[key]["time"].append(median_exec_time)
        # print(f"{key} avg acc = {sum(data[key]['acc'])/NUM_TESTS:.4f}")

        key = f"zeroing-{error_rate}"
        data[key] = {}
        data[key]["acc"] = []
        data[key]["time"] = []
        for test in range(NUM_TESTS):
            exec_times = []

            model.eval()
            quantized_ssm.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in tqdm(test_loader, desc=f"Correction Testing {key} {test+1}/{NUM_TESTS}", leave=False):
                    x, y = x.to(device), y.to(device)
                    u = model._process_to_ssm(x)

                    Bz, L, input_dim = u.shape
                    u_t = u.permute(1, 0, 2)

                    model_state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
                    outputs = []

                    for t in range(L):
                        start = time.perf_counter()

                        ut = u_t[t]
                        model_state = model.classifier.ssm.hidden_update(model_state, ut, inject_type="weight", error_rate=error_rate)

                        # model_state = nan_checker(model_state)

                        # mask = running_stats[t].cantelli(model_state)
                        # model_state[mask] = 0

                        y_t = model.classifier.ssm.get_output(model_state, ut)

                        mask = running_stats[t].cantelli(y_t)
                        y_t[mask] = 0

                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_times.append(end - start)

                    output = torch.stack(outputs, dim=1)
                    output = model.classifier.ssm.activation(output)
                    output = model.classifier.ssm.norm(output)
                    output = output.transpose(1, 2).contiguous()
                    logits = model.classifier.fc(output)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            test_acc = correct / total
            median_exec_time = stats.median(exec_times) * 1000

            data[key]["acc"].append(test_acc)
            data[key]["time"].append(median_exec_time)
        print(f"{key} avg acc = {sum(data[key]['acc'])/NUM_TESTS:.4f}")

        # key = f"backup-{error_rate}"
        # data[key] = {}
        # data[key]["acc"] = []
        # data[key]["time"] = []
        # for test in range(NUM_TESTS):
        #     exec_times = []

        #     model.eval()
        #     quantized_ssm.eval()
        #     correct, total = 0, 0
        #     with torch.no_grad():
        #         for x, y in tqdm(test_loader, desc=f"Correction Testing {key} {test+1}/{NUM_TESTS}", leave=False):
        #             x, y = x.to(device), y.to(device)
        #             u = model._process_to_ssm(x)

        #             Bz, L, input_dim = u.shape
        #             u_t = u.permute(1, 0, 2)

        #             model_state = model.classifier.ssm.h0.unsqueeze(0).expand(Bz, -1).contiguous()
        #             outputs = []

        #             for t in range(L):
        #                 start = time.perf_counter()

        #                 ut = u_t[t]
        #                 quantized_state = quantized_ssm.hidden_update(model_state, ut, inject_type="weight", error_rate=error_rate)
        #                 model_state = model.classifier.ssm.hidden_update(model_state, ut, inject_type="weight", error_rate=error_rate)

        #                 model_state = nan_checker(model_state)
        #                 quantized_state = nan_checker(quantized_state)

        #                 mask = running_stats[t].cantelli(model_state)
        #                 model_state[mask] = quantized_state[mask]

        #                 y_t = model.classifier.ssm.get_output(model_state, ut)
        #                 outputs.append(y_t)

        #                 end = time.perf_counter()
        #                 exec_times.append(end - start)

        #             output = torch.stack(outputs, dim=1)
        #             output = model.classifier.ssm.activation(output)
        #             output = model.classifier.ssm.norm(output)
        #             output = output.transpose(1, 2).contiguous()
        #             logits = model.classifier.fc(output)
        #             preds = logits.argmax(dim=1)
        #             correct += (preds == y).sum().item()
        #             total += y.size(0)

        #     test_acc = correct / total
        #     median_exec_time = stats.median(exec_times) * 1000

        #     data[key]["acc"].append(test_acc)
        #     data[key]["time"].append(median_exec_time)
        # print(f"{key} avg acc = {sum(data[key]['acc'])/NUM_TESTS:.4f}")

    output_path = f"cifar100_stats_weight_zeroing.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)