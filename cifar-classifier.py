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
import torch.nn.utils.prune as prune
from matplotlib import pyplot as plt

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

    def forward(self, u, inject=False, error_rate=1e-3):
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

            if t == 7 and inject:
                x_t, _ = flip_bits(x_t, error_rate=1)

            state = x_t

            # y_t = C @ state + D @ u_t
            y_t = state @ self.C.t() + (ut @ self.D.t())  # (B, output_dim)
            outputs.append(y_t)

        # stack => (L, B, output_dim) -> permute to (B, output_dim, L)
        Y = torch.stack(outputs, dim=0).permute(1, 2, 0).contiguous()
        return Y

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

def load_and_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu() else 'cpu')
    print("Device:", device)

    _, testloader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, augment=False)
    model = CIFAR_SSM_Classifier(ssm_state_dim=args.ssm_state_dim, ssm_out_dim=args.ssm_out_dim)
    model = model.to(device)
    model.load_state_dict(torch.load(args.save_path, map_location=device)['model_state'])

    eval_criterion = nn.CrossEntropyLoss()
    clean_loss, clean_acc = evaluate(model, testloader, device, eval_criterion,
                                     compute_grad=True, desc='Clean Eval (with grad compute)')
    print(f"Clean Eval - Acc: {clean_acc:.2f}")

    faulty_loss, faulty_acc = evaluate(model, testloader, device, eval_criterion,
                                       inject=True, error_rate=1e-3, correct_error=False, desc='Faulty Eval (no correction)')
    print(f"Faulty Eval - Acc: {faulty_acc:.2f}")

    corrected_loss, corrected_acc = evaluate(model, testloader, device, eval_criterion,
                                            inject=True, error_rate=1e-3, correct_error=True, desc='Corrected Eval')
    print(f"Corrected Eval - Acc: {corrected_acc:.2f}")

def load_and_get_correlation(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu() else 'cpu')
    print("Device:", device)

    _, testloader = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers, augment=False)
    model = CIFAR_SSM_Classifier(ssm_state_dim=args.ssm_state_dim, ssm_out_dim=args.ssm_out_dim)
    model = model.to(device)
    model.load_state_dict(torch.load(args.save_path, map_location=device)['model_state'])

    num_classes = 10
    activations_by_class = {i: [] for i in range(num_classes)}

    model.eval()
    with torch.no_grad():
        pbar = tqdm(testloader, desc='Getting Correlation', leave=False)
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            # if inject:
            #     print(f"Evaluating with fault injection (error rate={error_rate}, correct={correct})")
            outputs, activations = model(images)
            activations = activations.cpu()
            targets = targets.cpu()

            for cls in range(num_classes):
                mask = (targets == cls)
                if mask.any():
                    activations_by_class[cls].append(activations[mask])
    
    for cls in range(num_classes):
        if len(activations_by_class[cls]) > 0:
            activations_by_class[cls] = torch.cat(activations_by_class[cls], dim=0)  # shape [N_cls, num_neurons]
        else:
            activations_by_class[cls] = torch.empty(0)

    corr_by_class = {}

    for cls in range(num_classes):
        acts = activations_by_class[cls].numpy()  # shape [N_cls, num_neurons]
        
        if acts.shape[0] > 1:
            # Transpose so neurons are variables, samples are along axis 1
            corr = np.corrcoef(acts, rowvar=False)  # shape [num_neurons, num_neurons]
            corr_by_class[cls] = corr
        else:
            corr_by_class[cls] = None  # not enough samples

    for cls in range(num_classes):
        if corr_by_class[cls] is not None:
            plt.imshow(corr_by_class[cls], cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title(f"Neuron correlation matrix for class {cls}")
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
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

    # load_and_test(args)

    load_and_get_correlation(args)
