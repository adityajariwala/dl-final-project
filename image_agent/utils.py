import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_raw, transform=torchvision.transforms.ToTensor(), min_size=20):
        self.images = []
        self.puck_binary = []
        self.coordinates = []
        self.z = []

        for data in dataset_raw:
            self.images.append(data[0])
            self.puck_binary.append(1.0 if data[1][0] else 0.0)
            self.coordinates.append(data[1][1])
            self.z.append(data[1][2])

        self.transform = transform
        self.min_size = min_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_raw = self.images[idx]
        puck_binary = self.puck_binary[idx]
        coordinates = self.coordinates[idx]
        z = self.z[idx]

        img = Image.fromarray(img_raw)
        if self.transform is not None:
            img = self.transform(img)

        return img, puck_binary, coordinates, z


def load_detection_data(dataset_raw, num_workers=4, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_raw, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

