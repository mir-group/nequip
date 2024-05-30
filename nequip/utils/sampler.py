# inspired by
# https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py

from torch.utils.data import Sampler


class DistributedValidationSampler(Sampler):
    r"""
    Unlike DistributedSampler, this does NOT add extra copies of data to
    round to a multiple of `num_replicas`. This is important for getting
    validation right.

    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.total_size = len(self.dataset)  # true value without extra samples
        self.num_samples = len(
            range(self.rank, self.total_size, self.num_replicas)
        )  # true value without extra samples
        self.shuffle = False

    def __iter__(self):
        return iter(range(self.rank, self.total_size, self.num_replicas))

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch
