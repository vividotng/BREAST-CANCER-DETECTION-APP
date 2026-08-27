import torch
import torch.nn as nn
from torchvision import models


class MammogramResNetV10(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        # ResNet18
        resnet = models.resnet18(weights=None)

        # Grayscale input: 1 channel
        resnet.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Two classes: BENIGN / MALIGNANT
        resnet.fc = nn.Linear(
            resnet.fc.in_features,
            num_classes
        )

        # Put ResNet layers directly into this model
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = resnet.avgpool
        self.fc = resnet.fc

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


def load_model(model_path, device):

    model = MammogramResNetV10(num_classes=2)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model