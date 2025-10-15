import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ============ Model Components ============

class SSM(nn.Module):
    def __init__(self, d_model, d_state=256):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, d_model)
        B, L, D = x.shape
        state = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            u_t = x[:, t, :]
            state = state @ self.A.T + u_t @ self.B.T
            y_t = state @ self.C.T
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        return self.norm(self.activation(y))


def patchify(images, patch_size):
    # images: (B, C, H, W)
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0
    ph = H // patch_size
    pw = W // patch_size
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(B, ph * pw, C * patch_size * patch_size)
    return patches


class PatchSSMClassifier(nn.Module):
    def __init__(self, patch_size=4, d_model=128, d_state=256, n_classes=100):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        self.input_proj = nn.Linear(patch_dim, d_model)
        self.ssm = SSM(d_model=d_model, d_state=d_state)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, images):
        # images: (B, 3, 32, 32)
        patches = patchify(images, self.patch_size)  # (B, L, patch_dim)
        x = self.input_proj(patches)                 # (B, L, d_model)
        x = self.ssm(x)                              # (B, L, d_model)
        x = x.transpose(1, 2)                        # (B, d_model, L)
        x = self.pool(x).squeeze(-1)                 # (B, d_model)
        logits = self.fc(x)
        return logits

# ============ Training Utilities ============

def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_acc += accuracy(outputs, labels) * images.size(0)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Eval", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total_acc += accuracy(outputs, labels) * images.size(0)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

# ============ Main ============

def main():
    # ----- Hyperparameters -----
    batch_size = 128
    lr = 1e-3
    epochs = 50
    patch_size = 4
    d_model = 128
    d_state = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Dataset & DataLoader -----
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    # ----- Model, Loss, Optimizer -----
    model = PatchSSMClassifier(
        patch_size=patch_size,
        d_model=d_model,
        d_state=d_state,
        n_classes=100
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ----- Training Loop -----
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Test Loss: {test_loss:.4f} Acc: {test_acc*100:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), "patch_ssm_cifar100.pth")
    print("✅ Training complete. Model saved as patch_ssm_cifar100.pth")

if __name__ == "__main__":
    main()
