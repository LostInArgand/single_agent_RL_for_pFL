import torch
from torch import nn

class CIFAR10CNN(nn.Module):
    """
    Simple CNN with 3 convolutional blocks + classifier.
    RL agent will choose how many of these "layers" (blocks + classifier)
    to train for each client in each round.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # define blocks explicitly as list
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 32x16x16
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 64x8x8
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),    # 128x4x4
        )

        self.blocks = nn.ModuleList([
            self.block1,
            self.block2,
            self.block3,
        ])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = self.classifier(x)
        return x

    @property
    def num_layers_for_rl(self):
        # 3 blocks + 1 classifier = 4 "layers" for RL to choose from
        return len(self.blocks) + 1


if __name__ == "__main__":
    print("Creating model...")
    model = CIFAR10CNN(num_classes=10)
    print(model)

    print("\nTesting forward pass...")

    # create dummy CIFAR10 batch (batch_size=4)
    dummy_input = torch.randn(4, 3, 32, 32)

    output = model(dummy_input)

    print("Output shape:", output.shape)
    print("num_layers_for_rl:", model.num_layers_for_rl)

    # sanity check
    assert output.shape == (4, 10), "Output shape is incorrect!"

