import pytest
import itertools

import torch

from nequip.data import PartialSampler

from test_dataloader import npz_dataset, NPZ_DATASET_FIXTURE_N_FRAMES  # noqa


@pytest.fixture(params=[True, False], scope="module")
def shuffle(request) -> bool:
    return request.param


@pytest.fixture(
    params=[None, 1, 2, 5, 7, NPZ_DATASET_FIXTURE_N_FRAMES], scope="function"
)
def sampler(request, npz_dataset, shuffle) -> PartialSampler:  # noqa: F811
    return PartialSampler(
        data_source=npz_dataset,
        shuffle=shuffle,
        num_samples_per_epoch=request.param,
        generator=torch.Generator().manual_seed(0),
    )


def test_partials_add_up(sampler: PartialSampler):
    """Confirm that full data epochs are (random permutations of) the list of all dataset indexes"""
    seq = []
    for epoch_i in range(2 * sampler.num_samples_total + 1):
        sampler.set_epoch(epoch_i)
        seq.extend(iter(sampler))

    seq = [int(e) for e in seq]

    if sampler.shuffle:
        # make sure we've at least hit every frame once
        assert set(seq) == set(range(sampler.num_samples_total))
        # then go through it by dataset epochs
        i = 0
        while True:
            data_epoch_idexes = seq[i : i + sampler.num_samples_total]
            if len(data_epoch_idexes) == 0:
                break
            if len(data_epoch_idexes) == sampler.num_samples_total:
                # it should be a random permutation
                assert set(data_epoch_idexes) == set(range(sampler.num_samples_total))
            elif len(data_epoch_idexes) < sampler.num_samples_total:
                # we hae a partial dataset epoch at the end
                assert set(data_epoch_idexes) <= set(range(sampler.num_samples_total))
                assert len(set(data_epoch_idexes)) == len(data_epoch_idexes)
            else:
                assert False
            i += sampler.num_samples_total
    else:
        # make sure its a repeating sequence of aranges
        assert (
            seq
            == list(
                itertools.chain(
                    *[
                        range(sampler.num_samples_total)
                        for _ in range(sampler._epoch + 2)
                    ]
                )
            )[: len(seq)]
        )


def test_distributed():
    world_size = 1
    num_samples_per_epoch = world_size * 17
    n_extra = 23
    total_num_samples = num_samples_per_epoch + n_extra
    dataset = range(total_num_samples)  # can mock this object for this test
    samplers = [
        PartialSampler(
            data_source=dataset,
            shuffle=True,
            num_samples_per_epoch=num_samples_per_epoch,
            generator=torch.Generator().manual_seed(42),
            rank=rank,
            world_size=world_size,
        )
        for rank in range(world_size)
    ]
    for s in samplers:
        s.set_epoch(0)
    assert (
        len(
            set(range(total_num_samples))
            - set(itertools.chain(*[(int(e) for e in samp) for samp in samplers]))
        )
        == n_extra
    )


def test_epoch_count(sampler: PartialSampler):
    with pytest.raises(AssertionError):
        list(iter(sampler))
    sampler.set_epoch(0)
    assert sampler._epoch == 0
    assert sampler._prev_epoch is None
    list(iter(sampler))
    assert sampler._prev_epoch == 0
    with pytest.raises(AssertionError):
        list(iter(sampler))
    sampler.set_epoch(1)
    list(iter(sampler))
    assert sampler._epoch == 1
    assert sampler._prev_epoch == 1  # since that's the prev epoch we've just completed
