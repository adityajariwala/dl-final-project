import torch
import torch.nn.functional as F


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)
    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]


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
            # torch.nn.MaxPool2d(2),
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
    if name == 'model_puck.pt' or name == 'model_unified.pt':
        r = torch.load(path.join(path.dirname(path.abspath(__file__)), name))
    else:
        r = PuckDetector()
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    return r
