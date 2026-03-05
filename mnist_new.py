import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import wandb
import json

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
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L, D = x.shape
        device = x.device
        state = torch.zeros(B, self.d_state, device=device)

        outputs = []
        for t in range(L):
            u_t = x[:, t, :]

            A = self.A
            x_t = state @ A.T + u_t @ self.B.T

            state = x_t
            y_t = state @ self.C.T
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        activation = self.activation(y)
        norm = self.norm(activation)
        return torch.nan_to_num(norm, nan=0.0, posinf=1e6, neginf=-1e6)

    def single_step(self, state, u, inject=False, error_rate=0):
        if state is None:
            state = torch.zeros(u.size(0), self.d_state, device=u.device)
        x_next = state @ self.A.T + u @ self.B.T
        if inject:
            x_next, _ = flip_bits(x_next.detach(), error_rate=error_rate)
        y = x_next @ self.C.T
        return x_next, y
    
    def hidden_update(self, x, u, inject_type="none", error_rate=0):
        A, B = self.A, self.B

        if inject_type == "weight":
            A = flip_bits(A.detach(), error_rate=error_rate)
            B = flip_bits(B.detach(), error_rate=error_rate)

        left_term = F.linear(x, A)
        right_term = F.linear(u, B)
        
        if inject_type == "output":
            left_term = flip_bits(left_term.detach(), error_rate=error_rate)
            right_term = flip_bits(right_term.detach(), error_rate=error_rate)

        x_next = left_term + right_term
        return x_next
    
    def get_output(self, state, inject_type="none", error_rate=1e-3):
        C = self.C

        if inject_type == "weight":
            C = flip_bits(C.detach(), error_rate=error_rate)
        
        output = state @ C.T

        if inject_type == "output":
            output = flip_bits(output.detach(), error_rate=error_rate)

        return output

# ---------- Classifier Model ----------
class SSMClassifier(nn.Module):
    def __init__(self, d_model=128, d_state=64, n_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(28, d_model)
        self.ssm = SSM(d_model=d_model, d_state=d_state)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, error_rate=1e-4, compute_grad=False, inject=False, correct=False):
        x = x.squeeze(1)                # (B, 28, 28)
        x = self.input_proj(x)          # (B, 28, d_model)
        x = self.ssm(x, error_rate=error_rate, inject=inject)
        
        B, L, D = x.shape

        x = x.transpose(1, 2).contiguous()      # (B, d_model, 28)
        x = self.pool(x).squeeze(-1)    # (B, d_model)

        return self.fc(x)

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

class StateSpaceWithAdapter(nn.Module):
    def __init__(self, A, B, C):
        super().__init__()
        self.A = ParamWithAdapter(A)
        self.B = ParamWithAdapter(B)
        self.C = ParamWithAdapter(C)

    def forward(self, x, u, use_mask=False):
        x_next = self.hidden_update(x, u, use_mask=use_mask)
        y = self.get_output(x_next, use_mask=use_mask)
        return x_next, y
    
    def hidden_update(self, x, u, use_mask=False):
        left_term = self.A(x, use_mask=use_mask)
        right_term = self.B(u, use_mask=use_mask)
        x_next = left_term + right_term
        return x_next

    def get_output(self, x, use_mask=False):
        y = self.C(x, use_mask=use_mask)
        return y

    def get_sparsity(self, use_mask=False, effective=True):
        total_zeros = 0
        total_params = 0
        
        for layer in [self.A, self.B, self.C]:
            if effective:
                w = layer.get_weight(use_mask=use_mask)
            else:
                w = layer.param.data
            
            total_zeros += (w == 0).sum().item()
            total_params += w.numel()
        
        return total_zeros / total_params

class QuantizedSSM(nn.Module):
    def __init__(self, A, B, C, N=8):
        super().__init__()

        self.N = N

        s_A = torch.max(torch.abs(A)) / (2 ** (N-1) - 1)
        A_bar = torch.clamp(torch.round(A / s_A), -2**(N-1), 2**(N-1)-1)
        s_B = torch.max(torch.abs(B)) / (2 ** (N-1) - 1)
        B_bar = torch.clamp(torch.round(B / s_B), -2**(N-1), 2**(N-1)-1)
        s_C = torch.max(torch.abs(C)) / (2 ** (N-1) - 1)
        C_bar = torch.clamp(torch.round(C / s_C), -2**(N-1), 2**(N-1)-1)

        self.d_state = A.shape[0]
        self.d_model = C.shape[0]
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(self.d_model)

        # print(f"A: {A}\nA_bar: {A_bar}\nA_new:{A_bar * s_A}")
        # print(f"B: {B}\nB_bar: {B_bar}\nB_new:{B_bar * s_B}")
        # print(f"C: {C}\nC_bar: {C_bar}\nC_new:{C_bar * s_C}")

        self.A = nn.Parameter(A_bar)
        self.B = nn.Parameter(B_bar)
        self.C = nn.Parameter(C_bar)
        self.s_A = s_A
        self.s_B = s_B
        self.s_C = s_C

        self.tau = float('inf')
        self.checksum_norms = []
        self.active = False

        self.x_min = float('-inf')
        self.x_max = float('inf')

    def forward(self, x, u):
        x_next = self.hidden_update(x, u)
        y = self.get_output(x_next)
        return x_next, y
    
    def hidden_update(self, x, u, inject_type="none", error_rate=0):
        A_q = self.A.detach().clone()
        B_q = self.B.detach().clone()

        # memory access errors
        if inject_type == "weight":
            A_q = torch.clamp(
                flip_bits(A_q, error_rate=error_rate),
                -2**(self.N-1), 2**(self.N-1)-1
            )
            B_q = torch.clamp(
                flip_bits(B_q, error_rate=error_rate),
                -2**(self.N-1), 2**(self.N-1)-1
            )

        # dequantize weights
        A = A_q * self.s_A
        B = B_q * self.s_B

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

    def get_output(self, x):
        # dequantize weights
        C = self.C * self.s_C

        # get output
        y = F.linear(x, C)

        return y
    
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
        
        for layer in [self.A, self.B, self.C]:
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

def evaluate_child(model, child, test_loader, use_mask=False):
    model.eval()
    child.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing Child SSM Model", leave=False):
            x, y = x.to(device), y.to(device)
            x = x.squeeze(1)                # (B, 28, 28)
            x = model.input_proj(x)          # (B, 28, d_model)
            B, L, D = x.shape
            
            state = torch.zeros(B, model.ssm.d_state, device=device)
            outputs = []
            
            for t in range(L):
                u_t = x[:, t, :]
                if isinstance(child, StateSpaceWithAdapter):
                    state, y_t = child(state, u_t, use_mask=use_mask)
                else:
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
    return test_acc

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

    A = model.ssm.A.data.clone()
    B = model.ssm.B.data.clone()
    C = model.ssm.C.data.clone()

    desired_quantized_model = f"quantized_ssm_structured.pth"
    child = StateSpaceWithAdapter(A, B, C).to(device)

    params = {"A": child.A.param, "B": child.B.param, "C": child.C.param}

    base_acc = evaluate_child(model, child, test_loader)
    acc_threshold = base_acc - 0.1

    print(f"Initial Accuracy: {base_acc*100:.2f}%")

    lr = 5e-6
    num_epochs = 10
    lambda_reg = 1e-8

    for name, param in child.named_parameters():
        param.requires_grad = ("threshold" not in name and "quant" not in name)

    optimizer = torch.optim.Adam(
        [p for p in child.parameters() if p.requires_grad],
        lr=lr
    )
    criterion = nn.MSELoss()

    # wandb.init(
    #     project="quantized-ssm",
    #     config={
    #         "epochs": num_epochs,
    #         "lambda_reg": lambda_reg,
    #     }
    # )

    # wandb.define_metric("train_step")
    # wandb.define_metric("prune_step")
    # wandb.define_metric("epoch")
    # wandb.define_metric("prune_epoch")
    # wandb.define_metric("quant_step")

    # wandb.define_metric("train_step/*", step_metric="train_step")
    # wandb.define_metric("train/*", step_metric="epoch")
    # wandb.define_metric("prune/*", step_metric="prune_epoch")
    # wandb.define_metric("prune_step/*", step_metric="prune_step")
    # wandb.define_metric("quant_step/*", step_metric="quant_step")

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

                x = x.squeeze(1)                # (B, 28, 28)
                x = model.input_proj(x)          # (B, 28, d_model)
                B, L, D = x.shape

                state = torch.zeros(B, model.ssm.d_state, device=device)
                parent_state = torch.zeros(B, model.ssm.d_state, device=device)

                distill_loss = 0.0

                for t in range(L):
                    u_t = x[:, t, :]
                    state, y_t = child(state, u_t)
                    with torch.no_grad():
                        parent_state, parent_y_t = model.ssm.single_step(parent_state, u_t)

                    distill_loss += criterion(state, parent_state.detach()) + criterion(y_t, parent_y_t.detach())

                distill_loss /= L
                l1_A = child.A.param.abs().sum()
                l1_B = child.B.param.abs().sum()
                l1_C = child.C.param.abs().sum()
                l1_reg = l1_A + l1_B + l1_C

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
                # })

                global_step += 1

            test_acc = evaluate_child(model, child, test_loader)
            print(f"Child SSM Model Test Acc={test_acc:.4f}")

            if do_prune and test_acc < acc_threshold:
                do_prune = False
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-4
            elif not do_prune and test_acc > acc_threshold:
                do_prune = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-6

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
        num_epochs_prune = 10
        lr_threshold = 1e-2
        lambda_sparse = 1e-1

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

        global_step = 0
        for epoch in range(num_epochs_prune):
            child.train()
            model.eval()
            for x, y in tqdm(train_loader, desc=f"Child SSM Pruning Epoch {epoch+1}/{num_epochs_prune}", leave=False):
                x, y = x.to(device), y.to(device)

                x = x.squeeze(1)                # (B, 28, 28)
                x = model.input_proj(x)          # (B, 28, d_model)
                B, L, D = x.shape

                state = torch.zeros(B, model.ssm.d_state, device=device)
                parent_state = torch.zeros(B, model.ssm.d_state, device=device)

                distill_loss = 0.0

                for t in range(L):
                    u_t = x[:, t, :]
                    state, y_t = child(state, u_t, use_mask=True)
                    with torch.no_grad():
                        parent_state, parent_y_t = model.ssm.single_step(parent_state, u_t)

                    distill_loss += criterion(state, parent_state.detach()) + criterion(y_t, parent_y_t.detach())
        
                distill_loss /= L
                
                sparsity_push = -lambda_sparse * (
                    child.A.adapter._threshold +
                    child.B.adapter._threshold +
                    child.C.adapter._threshold
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
                child.A.adapter.threshold.item(),
                child.B.adapter.threshold.item(),
                child.C.adapter.threshold.item()
            ]

            print(f"Epoch {epoch+1:02d} | "
                f"Masked Acc: {current_acc*100:.2f}% | "
                f"Thresholds: {[round(t,4) for t in thresholds]}")

            # Prevent over-pruning
            if current_acc < target_acc and (epoch+1) < num_epochs_prune:
                print("⚠ Accuracy dropped too much — soft rollback")
                with torch.no_grad():
                    child.A.adapter._threshold -= 0.001
                    child.B.adapter._threshold -= 0.001
                    child.C.adapter._threshold -= 0.001

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

    # quantize the parent model
    # model.ssm = QuantizedSSM(
    #     model.ssm.A.data,
    #     model.ssm.B.data,
    #     model.ssm.C.data,
    #     N=4
    # ).to(device)

    # PTQ
    quantized_ssm = QuantizedSSM(
        child.A.get_weight(use_mask=True),
        child.B.get_weight(use_mask=True),
        child.C.get_weight(use_mask=True),
        N=4
    ).to(device)
    acc = evaluate_child(model, quantized_ssm, test_loader)
    print(f"Accuracy of Quantized Model: {acc*100:.2f}%")
    sparsity = quantized_ssm.get_sparsity()
    print(f"Sparsity: {sparsity*100:.2f}%")
        
    # test the quantized model and collect gradients
    grad_collector_x = Correction_Module_dense(k=4)
    grad_collector_y = Correction_Module_dense(k=4)

    model.eval()
    quantized_ssm.eval()
    correct, total = 0, 0

    x_min_running = float('inf')
    x_max_running = float('-inf')

    running_stats = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Collecting Statistics", leave=False):
            x, y = x.to(device), y.to(device)
            x = x.squeeze(1)                # (B, 28, 28)
            x = model.input_proj(x)          # (B, 28, d_model)
            B, L, D = x.shape

            if not running_stats:
                running_stats = [RunningStats() for _ in range(L)]

            state = torch.zeros(B, quantized_ssm.A.size(0), device=device)
            model_state = torch.zeros(B, model.ssm.d_state, device=device)
            
            outputs = []
            for t in range(L):
                u_t = x[:, t, :]
                state, y_t = quantized_ssm(state, u_t)
                model_state = model.ssm.hidden_update(model_state, u_t)

                # grad_collector_x.compute_grad(model_state, f"ssm_state_{t}")
                
                # running_stats[t].update(model_state)

                # Compute batch quantiles
                batch_flat = state.view(-1)
                batch_x_min = batch_flat.quantile(0.01).item()
                batch_x_max = batch_flat.quantile(0.99).item()

                # Update running x_min/x_max
                x_min_running = min(x_min_running, batch_x_min)
                x_max_running = max(x_max_running, batch_x_max)

                model_output = model.ssm.get_output(model_state)
                running_stats[t].update(model_output)                

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

    print(f"x_min={x_min_running:.4f}, x_max={x_max_running:.4f}")
    quantized_ssm.set_bounds(x_min_running, x_max_running)


    tau = quantized_ssm.compute_tau()
    print(f"tau: {tau:.4f}")

    # get a new random split
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)

    error_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
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
        #             x = x.squeeze(1)                # (B, 28, 28)
        #             x = model.input_proj(x)          # (B, 28, d_model)
        #             B, L, D = x.shape
        #             model_state = torch.zeros(B, model.ssm.d_state, device=device)
        #             outputs = []

        #             for t in range(L):
        #                 start = time.perf_counter()

        #                 u_t = x[:, t, :]
        #                 # quantized_state = quantized_ssm.hidden_update(quantized_state, u_t, inject=True, inject_type="output", error_rate=error_rate)
        #                 model_state = model.ssm.hidden_update(model_state, u_t, inject_type="output", error_rate=error_rate)

        #                 # model_state = nan_checker(model_state)
        #                 # quantized_state = nan_checker(quantized_state)

        #                 # model_state = grad_collector(model_state, f"ssm_state_{t}")
        #                 # diff = torch.abs(model_state - quantized_state)
        #                 # mask = torch.abs(diff - mu_diff) > k * std_diff
        #                 # model_state[mask] = quantized_state[mask]

        #                 y_t = model.ssm.get_output(model_state)
        #                 outputs.append(y_t)

        #                 end = time.perf_counter()
        #                 exec_times.append(end - start)

        #             output = torch.stack(outputs, dim=1)
        #             output = model.ssm.activation(output)
        #             output = model.ssm.norm(output)
        #             x = output.transpose(1, 2).contiguous()      # (B, d_model, 28)
        #             x = model.pool(x).squeeze(-1)    #
        #             logits = model.fc(x)
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
                    x = x.squeeze(1)                # (B, 28, 28)
                    x = model.input_proj(x)          # (B, 28, d_model)
                    B, L, D = x.shape
                    model_state = torch.zeros(B, model.ssm.d_state, device=device)
                    outputs = []

                    for t in range(L):
                        start = time.perf_counter()

                        u_t = x[:, t, :]
                        # quantized_state = quantized_ssm.hidden_update(quantized_state, u_t, inject=True, inject_type="output", error_rate=error_rate)
                        model_state = model.ssm.hidden_update(model_state, u_t, inject_type="output", error_rate=error_rate)

                        # model_state = nan_checker(model_state)
                        # quantized_state = nan_checker(quantized_state)

                        # mask = running_stats[t].cantelli(model_state)
                        # model_state[mask] = 0

                        # model_state = grad_collector_x(model_state, f"ssm_state_{t}")
                        # diff = torch.abs(model_state - quantized_state)
                        # mask = torch.abs(diff - mu_diff) > k * std_diff
                        # model_state[mask] = quantized_state[mask]

                        y_t = model.ssm.get_output(model_state)
                        
                        mask = running_stats[t].cantelli(y_t)
                        y_t[mask] = 0

                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_times.append(end - start)

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
        #             x = x.squeeze(1)                # (B, 28, 28)
        #             x = model.input_proj(x)          # (B, 28, d_model)
        #             B, L, D = x.shape
        #             model_state = torch.zeros(B, model.ssm.d_state, device=device)
        #             outputs = []

        #             for t in range(L):
        #                 start = time.perf_counter()

        #                 u_t = x[:, t, :]
        #                 quantized_state = quantized_ssm.hidden_update(model_state, u_t, inject_type="output", error_rate=error_rate)
        #                 model_state = model.ssm.hidden_update(model_state, u_t, inject_type="output", error_rate=error_rate)

        #                 model_state = nan_checker(model_state)
        #                 quantized_state = nan_checker(quantized_state)

        #                 mask = running_stats[t].cantelli(model_state)
        #                 model_state[mask] = quantized_state[mask]

        #                 # model_state = grad_collector_x(model_state, f"ssm_state_{t}", replace_tensor=quantized_state)
                        
        #                 y_t = model.ssm.get_output(model_state)
        #                 outputs.append(y_t)

        #                 end = time.perf_counter()
        #                 exec_times.append(end - start)

        #             output = torch.stack(outputs, dim=1)
        #             output = model.ssm.activation(output)
        #             output = model.ssm.norm(output)
        #             x = output.transpose(1, 2).contiguous()      # (B, d_model, 28)
        #             x = model.pool(x).squeeze(-1)    #
        #             logits = model.fc(x)
        #             preds = logits.argmax(dim=1)
        #             correct += (preds == y).sum().item()
        #             total += y.size(0)

        #     test_acc = correct / total
        #     median_exec_time = stats.median(exec_times) * 1000

        #     data[key]["acc"].append(test_acc)
        #     data[key]["time"].append(median_exec_time)
        # print(f"{key} avg acc = {sum(data[key]['acc'])/NUM_TESTS:.4f}")

    output_path = "mnist_stats_output_zeroing.json"
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
        #             x = x.squeeze(1)                # (B, 28, 28)
        #             x = model.input_proj(x)          # (B, 28, d_model)
        #             B, L, D = x.shape
        #             model_state = torch.zeros(B, model.ssm.d_state, device=device)
        #             outputs = []

        #             for t in range(L):
        #                 start = time.perf_counter()

        #                 u_t = x[:, t, :]
        #                 # quantized_state = quantized_ssm.hidden_update(quantized_state, u_t, inject=True, inject_type="output", error_rate=error_rate)
        #                 model_state = model.ssm.hidden_update(model_state, u_t, inject_type="weight", error_rate=error_rate)

        #                 # model_state = nan_checker(model_state)
        #                 # quantized_state = nan_checker(quantized_state)

        #                 # model_state = grad_collector(model_state, f"ssm_state_{t}")
        #                 # diff = torch.abs(model_state - quantized_state)
        #                 # mask = torch.abs(diff - mu_diff) > k * std_diff
        #                 # model_state[mask] = quantized_state[mask]

        #                 y_t = model.ssm.get_output(model_state)
        #                 outputs.append(y_t)

        #                 end = time.perf_counter()
        #                 exec_times.append(end - start)

        #             output = torch.stack(outputs, dim=1)
        #             output = model.ssm.activation(output)
        #             output = model.ssm.norm(output)
        #             x = output.transpose(1, 2).contiguous()      # (B, d_model, 28)
        #             x = model.pool(x).squeeze(-1)    #
        #             logits = model.fc(x)
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
                    x = x.squeeze(1)                # (B, 28, 28)
                    x = model.input_proj(x)          # (B, 28, d_model)
                    B, L, D = x.shape
                    model_state = torch.zeros(B, model.ssm.d_state, device=device)
                    outputs = []

                    for t in range(L):
                        start = time.perf_counter()

                        u_t = x[:, t, :]
                        # quantized_state = quantized_ssm.hidden_update(quantized_state, u_t, inject=True, inject_type="output", error_rate=error_rate)
                        model_state = model.ssm.hidden_update(model_state, u_t, inject_type="weight", error_rate=error_rate)

                        model_state = nan_checker(model_state)
                        # quantized_state = nan_checker(quantized_state)

                        # mask = running_stats[t].cantelli(model_state)
                        # model_state[mask] = 0

                        y_t = model.ssm.get_output(model_state)
                        
                        mask = running_stats[t].cantelli(y_t)
                        y_t[mask] = 0

                        outputs.append(y_t)

                        end = time.perf_counter()
                        exec_times.append(end - start)

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
        #             x = x.squeeze(1)                # (B, 28, 28)
        #             x = model.input_proj(x)          # (B, 28, d_model)
        #             B, L, D = x.shape
        #             model_state = torch.zeros(B, model.ssm.d_state, device=device)
        #             outputs = []

        #             for t in range(L):
        #                 start = time.perf_counter()

        #                 u_t = x[:, t, :]
        #                 quantized_state = quantized_ssm.hidden_update(model_state, u_t, inject_type="weight", error_rate=error_rate)
        #                 model_state = model.ssm.hidden_update(model_state, u_t, inject_type="weight", error_rate=error_rate)

        #                 model_state = nan_checker(model_state)
        #                 quantized_state = nan_checker(quantized_state)

        #                 mask = running_stats[t].cantelli(model_state)
        #                 model_state[mask] = quantized_state[mask]

        #                 # model_state = grad_collector_x(model_state, f"ssm_state_{t}", replace_tensor=quantized_state)
                        
        #                 y_t = model.ssm.get_output(model_state)
        #                 outputs.append(y_t)

        #                 end = time.perf_counter()
        #                 exec_times.append(end - start)

        #             output = torch.stack(outputs, dim=1)
        #             output = model.ssm.activation(output)
        #             output = model.ssm.norm(output)
        #             x = output.transpose(1, 2).contiguous()      # (B, d_model, 28)
        #             x = model.pool(x).squeeze(-1)    #
        #             logits = model.fc(x)
        #             preds = logits.argmax(dim=1)
        #             correct += (preds == y).sum().item()
        #             total += y.size(0)

        #     test_acc = correct / total
        #     median_exec_time = stats.median(exec_times) * 1000

            # data[key]["acc"].append(test_acc)
            # data[key]["time"].append(median_exec_time)
        # print(f"{key} avg acc = {sum(data[key]['acc'])/NUM_TESTS:.4f}")

    output_path = "mnist_stats_weight_zeroing.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)