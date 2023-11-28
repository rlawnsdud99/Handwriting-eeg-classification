import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model_definition import EEGNet
import torch.nn as nn
import numpy as np
import random


# 랜덤 시드 고정 함수
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, x_train, y_train, x_val, y_val, num_epochs=None):
    # 데이터를 PyTorch Tensor로 변환
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    print(f"x_train_tensor: {x_train_tensor.shape}")
    # DataLoader 설정
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    print(model)
    # 학습
    seed_everything(42)
    for epoch in range(num_epochs):
        model.train()
        for i, (x_batch, y_batch) in enumerate(train_loader):  # 새로운 차원 추가
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                val_loss += criterion(outputs, y_batch).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(y_batch.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100.0 * correct / len(val_loader.dataset)

        print(
            f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

    torch.save(model.state_dict(), "trained_model.pth")

    return model
