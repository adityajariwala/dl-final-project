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


class BinaryClassifier(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(5, 7, kernel_size=3)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(6300, 1024)
        self.fc2 = torch.nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 32, 32]):
        super().__init__()

        conv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, 5, 2, 2), torch.nn.ReLU(True)]

        h, _conv = 3, []
        for c in channels:
            _conv += conv_block(c, h)
            h = c

        self._conv = torch.nn.Sequential(*_conv, torch.nn.Conv2d(h, 1, 1))
        # self.classifier = torch.nn.Linear(h, 2)
        # self.classifier = torch.nn.Conv2d(h, 1, 1)

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        return spatial_argmax(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))


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


# FOUND ON https://python.plainenglish.io/single-object-detection-with-pytorch-step-by-step-96430358ae9d
# CHANGE MODEL TO FIT OUR NEEDS
# THIS IS CURRENTLY PLAGARIZED FROM LINK ABOVE
# NOT SURE IF WE CAN CITE THE SOURCE, OR WE SHOULD JUST CHANGE THE NAME AND STRUCTURE
class FoveaNet(torch.nn.Module):
    def __init__(self, in_channels=3, first_output_channels=16):
        super().__init__()
        self.model = torch.nn.Sequential(
            ResBlock(in_channels, first_output_channels),
            torch.nn.MaxPool2d(2),
            ResBlock(first_output_channels, 2 * first_output_channels),
            torch.nn.MaxPool2d(2),
            ResBlock(2 * first_output_channels, 4 * first_output_channels),
            torch.nn.MaxPool2d(2),
            ResBlock(4 * first_output_channels, 8 * first_output_channels),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8 * first_output_channels, 16 * first_output_channels, kernel_size=3),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 16 * first_output_channels, 2)
        )

    def forward(self, x):
        return self.model(x)


class FoveaNetDist(torch.nn.Module):
    def __init__(self, in_channels=3, first_output_channels=16):
        super().__init__()
        self.model = torch.nn.Sequential(
            ResBlock(in_channels, first_output_channels),
            torch.nn.MaxPool2d(2),
            ResBlock(first_output_channels, 2 * first_output_channels),
            torch.nn.MaxPool2d(2),
            ResBlock(2 * first_output_channels, 4 * first_output_channels),
            torch.nn.MaxPool2d(2),
            ResBlock(4 * first_output_channels, 8 * first_output_channels),
            # torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8 * first_output_channels, 16 * first_output_channels, kernel_size=3),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 16 * first_output_channels, 1)
        )

    def forward(self, x):
        return self.model(x)


def save_model(model, name: str = 'det.pt'):
    from torch import save
    from os import path
    save(model, path.join(path.dirname(path.abspath(__file__)), name))
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name[:-3] + '.th'))


def load_model(name: str = 'model_puck.pt'):
    from torch import load
    from os import path
    if name == 'model_puck.pt' or name == 'model_coord.pt':
        r = torch.load(path.join(path.dirname(path.abspath(__file__)), name))
    else:
        r = BinaryClassifier()
        r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    return r
