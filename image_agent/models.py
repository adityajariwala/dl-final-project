import torch
import torch.nn.functional as F


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.base1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(True)
        )
        self.base2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        x = self.base1(x) + x
        x = self.base2(x)
        return x


class UnifiedCoordDist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Shared layers
        self.shared_layers = torch.nn.Sequential(
            ResBlock(3, 16),
            torch.nn.MaxPool2d(2),
            ResBlock(16, 32),
            torch.nn.MaxPool2d(2),
            ResBlock(32, 64),
            torch.nn.MaxPool2d(2)
        )

        # Branch for coordinate prediction
        self.coord_branch = torch.nn.Sequential(
            ResBlock(64, 128),
            torch.nn.Conv2d(128, 256, kernel_size=3),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 256, 2)
        )

        # Branch for distance prediction
        self.dist_branch = torch.nn.Sequential(
            ResBlock(64, 128),
            torch.nn.Conv2d(128, 256, kernel_size=3),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 256, 1)
        )

    def forward(self, x):
        x = self.shared_layers(x)
        coords = self.coord_branch(x)
        dist = self.dist_branch(x)
        return coords, dist


class PuckDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            ResBlock(3, 16),
            torch.nn.MaxPool2d(2),
            ResBlock(16, 32),
            torch.nn.MaxPool2d(2),
            ResBlock(32, 64),
            torch.nn.MaxPool2d(2),
            ResBlock(64, 128),
            torch.nn.Conv2d(128, 256, kernel_size=3),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 256, 2)
        )

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)


def save_model(model, name: str = 'det.pt'):
    from torch import save
    from os import path
    return save(model, path.join(path.dirname(path.abspath(__file__)), name))


def load_model(name: str = 'model_puck.pt'):
    from torch import load
    from os import path
    if name == 'model_puck.pt' or name == 'model_unified_new_25.pt':
        r = torch.load(path.join(path.dirname(path.abspath(__file__)), name))
    else:
        r = PuckDetector()
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    return r
