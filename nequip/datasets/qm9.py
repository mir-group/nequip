import torch
from typing import Tuple, Dict, Any, List, Callable, Union

from torch_geometric.datasets import QM9
from torch_geometric.nn import SchNet
from torch_geometric.data import DataLoader

from nequip.data import AtomicDataset


class QM9Dataset(AtomicDataset):
    def __init__(
        self,
        root="./",
        force_fixed_keys: List[str] = [],
        extra_fixed_fields: Dict[str, Any] = {},
    ):

        self.force_fixed_keys = force_fixed_keys
        self.extra_fixed_fields = extra_fixed_fields

        self.url = getattr(type(self), "URL", None)

        super().__init__(root=root)

        self.data, self.fixed_fields = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        pass

        # from torch_geometric.datasets import QM9
        # from torch_geometric.nn import SchNet

        # dataset = QM9(data_path)
        # atomref = dataset.atomref(target=target)

        # units = 1000 if target in [2, 3, 4, 6, 7, 8, 9, 10] else 1

        # _, datasets = SchNet.from_qm9_pretrained(data_path, dataset, target)

        # train_dataset, val_dataset, _test_dataset = datasets

        # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        # pos = []
        # elements = []
        # energies = []

        # for idx, data in enumerate(train_loader):

        #     if idx < n_train:

        #         pos.append(data.pos)
        #         elements.append(data.z)
        #         ref_energies = [atomref[el] for el in data.z]
        #         energies.append(data.y[:, target] - np.sum(ref_energies))

        #     else:
        #         break

        # for idx, data in enumerate(val_loader):

        #     if idx < n_val:

        #         pos.append(data.pos)
        #         elements.append(data.z)
        #         ref_energies = [atomref[el] for el in data.z]
        #         energies.append(data.y[:, target] - np.sum(ref_energies))

        #     else:
        #         break

        # if data_format == "numpy":
        #     raise ValueError("data_format has to be list for QM9 dataset")

        # return (elements, pos, None, None, energies)
