from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class Loader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, distributed, shuffle=True):

        if distributed:
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                sampler=sampler,
            )

        else:
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )

    def set_epoch(self, epoch):
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
