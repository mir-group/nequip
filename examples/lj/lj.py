"""Example implementation of a Lennard-Jones potential in the NequIP framework.

This serves as a basic example of how to write a NequIP framework model from scratch.
"""

from typing import Union

import torch

from nequip.nn import scatter
from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin, SequentialGraphNetwork, AtomwiseReduce


# First, we define a model module to do the actual computation:
class LennardJonesModule(GraphModuleMixin, torch.nn.Module):
    """NequIP model module implementing a Lennard-Jones potential term.

    See, for example, `lj/cut` in LAMMPS:
    https://docs.lammps.org/pair_lj.html

    Args:
        initial_epsilon: initial value of the epsilon parameter.
        initial_sigma: initial value of the sigma parameter.
        trainable: whether epsilon and sigma should be trainable.
            Default False.
    """

    def __init__(
        self,
        initial_epsilon: Union[float, torch.Tensor],
        initial_sigma: Union[float, torch.Tensor],
        trainable: bool = False,
        irreps_in=None,
    ) -> None:
        super().__init__()
        # We have to tell `GraphModuleMixin` what fields we expect in the input and output
        # and what their irreps will be. Having basic geometry information (positions and edges)
        # in the input is assumed.
        # Per-atom energy is a scalar, so 0e.
        self._init_irreps(irreps_out={AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"})
        self.trainable = trainable
        eps = torch.as_tensor(initial_epsilon)
        sigma = torch.as_tensor(initial_sigma)
        assert eps.ndim == sigma.ndim == 0, "epsilon and sigma must be scalars"
        if self.trainable:
            self.epsilon = torch.nn.Parameter(eps)
            self.sigma = torch.nn.Parameter(sigma)
        else:
            # buffers act like parameters, but are not trainable
            self.register_buffer("epsilon", eps)
            self.register_buffer("sigma", sigma)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        """Run the module.

        The module both takes and returns an `AtomicDataDict.Type` = `Dict[str, torch.Tensor]`.
        Keys that the module does not modify/add are expected to be propagated to the output unchanged.
        """
        # If they are not already present, compute and add the edge vectors and lengths to `data`:
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        # compute the LJ energy:
        lj_eng = (self.sigma / data[AtomicDataDict.EDGE_LENGTH_KEY]) ** 6.0
        lj_eng = torch.neg(lj_eng)
        lj_eng = lj_eng + lj_eng.square()
        # 2.0 because we do the slightly wastefull symmetric thing and let
        # ij and ji each contribute half
        # this avoids indexing out certain edges in the general case where
        # the edges are not ordered.
        lj_eng = (2.0 * self.epsilon) * lj_eng
        # assign halves to centers
        atomic_eng = scatter(
            lj_eng,
            # the edge indexes are of shape [2, n_edge];
            # edge_index[0] is the index of the central atom of each edge
            data[AtomicDataDict.EDGE_INDEX_KEY][0],
            dim=0,
            # dim_size says that even if some atoms have no edges, we still
            # want an output energy for them (it will be zero)
            dim_size=len(data[AtomicDataDict.POSITIONS_KEY]),
        )
        # NequIP defines standardized keys for typical fields:
        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atomic_eng
        return data


# then, we define a *model builder* function that builds an LJ energy model
# from this and other modules:
def LennardJonesPotential(config) -> SequentialGraphNetwork:
    # `from_parameters` builds a model containing each of these modules in sequence
    # from a configuration `config`
    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers={
            # LennardJonesModule will be built using options from `config`
            "lj": LennardJonesModule,
            # AtomwiseReduce will be built using the provided default parameters,
            # and also those from `config`.
            "total_energy_sum": (
                AtomwiseReduce,
                dict(
                    reduce="sum",
                    field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                    out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
                ),
            ),
        },
    )
