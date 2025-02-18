import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as T

import random
from tqdm import tqdm
from dataclasses import dataclass

import nn_modules as nnm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 1337
torch.manual_seed(seed)
print("Device {device}")
print("Seed {seed}")

@dataclass
class Config:
    batch_size : int = 32
    image_size : int = 32
    n_epochs : int = 5
    p_train_split : float = 0.9
    # normalize_shape : tuple = (0.5,)
    normalize_shape : tuple = (0.5, 0.5, 0.5)
    # num_channels : int = 1
    num_channels : int = 3
    lr : float = 1e-3

    do_small_sample : bool = True

class BasicBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride
    ):
        """
        identity = x
        conv1 ->
        bn1 ->
        relu ->

        conv2 ->
        bn2 ->
        downsample if necessary
        x = x + identity 
        """
        super().__init__()
        self.conv1 = nnm.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nnm.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nnm.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nnm.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nnm.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, padding=0
                ),
                nnm.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)

        return x

class Resnet18(nn.Module):
    def __init__(self, config, num_classes=10):
        """
        conv1 (k7, stride=2, padding3)
        bn1
        relu
        maxpool

        layer1 -> ... -> layer4

        avgpool
        flatten
        linear -> logits_{num_classes}
        """
        super().__init__()
        self.conv1 = nnm.Conv2d(
            in_channels=config.num_channels, out_channels=64,
            kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nnm.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nnm.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Maker our layers
        self.layer1 = self._make_layer(in_channels=64, out_channels=64, n_blocks=2, stride=1)
        self.layer2 = self._make_layer(in_channels=64, out_channels=128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(in_channels=128, out_channels=256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(in_channels=256, out_channels=512, n_blocks=2, stride=2)

        self.avgpool = nnm.AvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, n_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x) # (B, 512)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_test_model(config : Config):
    dataset = CIFAR10(
        root="~/projects/vision-transformer/data",
        download=False,
        transform=T.Compose([
            T.RandomCrop(config.image_size, padding=4),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(config.normalize_shape, config.normalize_shape),
        ])
    )
    
    if config.do_small_sample:
        indices = random.sample(range(len(dataset)), 5000)
        dataset = Subset(dataset, indices)
        print(len(dataset))

    train_split = int(config.p_train_split * len(dataset))
    train, test = random_split(dataset, [train_split, len(dataset) - train_split])
    traindl = DataLoader(train, batch_size=config.batch_size, shuffle=True)
    testdl = DataLoader(test, batch_size=config.batch_size, shuffle=False)

    model = Resnet18(config, num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs}")
        with tqdm(traindl, desc="Training") as pbar:
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            model.train()
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                train_total += inputs.size(0)
                pbar.set_postfix(loss=loss.item())

            train_epoch_loss = train_loss / train_total
            train_epoch_acc = train_correct / train_total
            tqdm.write(f"Train_Loss {train_epoch_loss:.4f} Train Acc {train_epoch_acc:.2f}")

        with tqdm(testdl, desc="Validation") as pbar:
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            model.eval()
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)
                pbar.set_postfix(loss=loss.item())

            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            tqdm.write(f"Val_Loss {val_epoch_loss:.4f} Val Acc {val_epoch_acc:.2f}")



def main():
    config = Config()
    train_test_model(config)

if __name__ == '__main__':
    main()
