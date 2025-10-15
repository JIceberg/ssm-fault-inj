import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# np.random.seed(42)
# torch.random.manual_seed(42)

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

        # Stats
        num_flips = 0
        num_detected = 0
        false_positives = 0

        if not self.training:
            A_col_sum = self.A.sum(dim=0, keepdim=True)
            A_aug = torch.cat([self.A, A_col_sum], dim=0) 

            B_col_sum = self.B.sum(dim=0, keepdim=True)
            B_aug = torch.cat([self.B, B_col_sum], dim=0) 

        outputs = []
        for t in range(L):
            u_t = x[:, t, :]

            if not self.training and inject:
                state_col_checksum = state.sum(dim=0, keepdim=True)  
                state_aug = torch.cat([state, state_col_checksum], dim=0)
                A_check = state_aug @ A_aug.T
                # assert torch.equal(A_check[:-1, :-1], state @ self.A.T)

                input_col_checksum = u_t.sum(dim=0, keepdim=True)
                input_aug = torch.cat([u_t, input_col_checksum], dim=0)
                B_check = input_aug @ B_aug.T
                # assert torch.equal(B_check[:-1, :-1], u_t @ self.B.T)

                x_t_check = A_check + B_check
                x_t_check, does_flip = flip_bits(x_t_check.detach(), error_rate=error_rate)
                x_t = x_t_check[:-1, :-1]

                if does_flip:
                    num_flips += 1

                expected_row_sum = x_t_check[:-1, -1]
                expected_col_sum = x_t_check[-1, :-1]
                row_sum = x_t.sum(dim=1)
                col_sum = x_t.sum(dim=0)

                col_check = torch.allclose(expected_col_sum, col_sum)
                row_check = torch.allclose(expected_row_sum, row_sum)
                detected = not col_check and not row_check

                if detected:
                    if does_flip:
                        num_detected += 1
                    else:
                        false_positives += 1
                    # x_t = state
            else:
                x_t = state @ self.A.T + u_t @ self.B.T

            state = x_t
            y_t = state @ self.C.T
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        return self.norm(self.activation(y)), num_flips, num_detected, false_positives


# ---------- Classifier Model ----------
class SSMClassifier(nn.Module):
    def __init__(self, d_model=128, d_state=64, n_classes=10):
        super().__init__()
        self.input_proj = nn.Linear(28, d_model)
        self.ssm = SSM(d_model=d_model, d_state=d_state)
        self.corr_dense = Correction_Module_dense()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x, error_rate=1e-4, compute_grad=False, inject=False, correct=False):
        x = x.squeeze(1)                # (B, 28, 28)
        x = self.input_proj(x)          # (B, 28, d_model)
        x, flips, detected, fp = self.ssm(x, error_rate=error_rate, inject=inject)
        x = x.transpose(1, 2)           # (B, d_model, 28)
        x = self.pool(x).squeeze(-1)    # (B, d_model)
        if not self.training:
            if compute_grad:
                self.corr_dense.compute_grad(x, "ssm_layer")
            if inject:
                x, _ = flip_bits(x, error_rate=error_rate)
                if correct:
                    x = self.corr_dense(x, "ssm_layer")
        return self.fc(x), flips, detected, fp


# ---------- Training Script ----------
def train_model(epochs=5, batch_size=64, lr=1e-3, device="cuda"):
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

    return model, test_loader


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, test_loader = train_model(epochs=5, device=device)

    # test
    model.eval()
    with torch.no_grad():
        flips = detected = fps = 0

        correct, total = 0, 0
        for x, y in tqdm(test_loader, desc=f"Compute Grads", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x, compute_grad=True)[0]
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        test_acc = correct / total
        print(f"Test Acc with no Errors={test_acc:.4f}")

        correct, total = 0, 0
        for x, y in tqdm(test_loader, desc=f"Inject Errors", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x, error_rate=1e-2, inject=True)[0]
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        test_acc = correct / total
        print(f"Test Acc with Errors, Uncorrected={test_acc:.4f}")

        correct, total = 0, 0
        for x, y in tqdm(test_loader, desc=f"Inject Errors", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x, error_rate=1e-2, inject=True, correct=True)[0]
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        test_acc = correct / total
        print(f"Test Acc with Errors, Corrected={test_acc:.4f}")
        

