import torch.nn as nn
import torch
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(self, num_classes, num_channels):
        super(EEGNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise = nn.Conv2d(16, 16, (1, 1), groups=16)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.25)

        self.separable = nn.Conv2d(16, 16, (1, 16))
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.25)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1536, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_last_layer=False):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        x = self.separable(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        if return_last_layer:
            return x  # 마지막 레이어 전까지의 출력 반환
        x = self.fc(x)
        x = self.softmax(x)

        return x


class EEG_LSTM(nn.Module):
    def __init__(self, input_size=32, hidden_size=64, num_layers=2, num_classes=11):
        super(EEG_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initializing hidden state for first input
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initializing cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


class Simple1DCNN(nn.Module):
    def __init__(self, num_channels=32, num_classes=10):
        super(Simple1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(
            128 * (501 // 4), 128
        )  # 501 is the length of time series, //4 due to two pooling layers
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(
            -1, 128 * (501 // 4)
        )  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TCNN(nn.Module):
    def __init__(self, num_channels=32, num_classes=10):
        super(TCNN, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(
            128, 256, kernel_size=5, stride=1, padding=2, dilation=2
        )  # 추가된 레이어
        self.fc1 = nn.Linear(256 * 20, 128)  # 업데이트된 출력 차원
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # 새로운 컨볼루션 레이어 적용
        x = x.view(-1, 256 * 20)  # 업데이트된 플래튼 차원
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
