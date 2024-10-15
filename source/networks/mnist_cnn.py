import torch.nn as nn

class MNISTConvNet(nn.Module):

    def __init__(self, p_drop: float = 0.0):
        super(MNISTConvNet, self).__init__()

        activation = nn.ReLU()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            activation,
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            activation,
            nn.Flatten(),
            nn.Linear(in_features=12 * 4**2, out_features=32),
            activation,
            nn.Dropout(p=p_drop, inplace=False),
            nn.Linear(in_features=32, out_features=32),
            activation,
            nn.Dropout(p=p_drop, inplace=False)
        )

        self.out = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        return self.out(self.projection(x))
