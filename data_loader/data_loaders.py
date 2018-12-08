from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset

from base import BaseDataLoader


class SampleDataLoader(BaseDataLoader):
    def __init__(
        self, data_dir, batch_size,
        shuffle, validation_split,
        num_workers, dataset_args={}
    ):
        self.data_dir = data_dir
        self.dataset = SampleDataSet(data_dir, **dataset_args)
        super(SampleDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class SampleDataSet(Dataset):
    def __init__(self, data_dir, training):
        self.dataset = datasets.MNIST(data_dir, train=training, download=True, transform=transforms.ToTensor())

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
