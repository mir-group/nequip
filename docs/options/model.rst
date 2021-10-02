Model
=====

Edge Basis
**********

Basic
-----

.. _r_max_option:

r_max
^^^^^
    | Type: float
    | Default: n/a

    The cutoff radius within which an atom is considered a neighbor.

irreps_edge_sh
^^^^^^^^^^^^^^
    | Type: :ref:`Irreps` or int
    | Default: n/a

    The irreps to use for the spherical harmonic projection of the edges.
    If an integer, specifies all spherical harmonics up to and including that integer as :math:`\ell_{\text{max}}`.
    If provided as explicit irreps, all multiplicities should be 1.

num_basis
^^^^^^^^^
    | Type: int
    | Default: ``8``

    The number of radial basis functions to use.

chemical_embedding_irreps_out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | Type: :ref:`Irreps`
    | Default: n/a

    The size of the linear embedding of the chemistry of an atom.

Advanced
--------

BesselBasis_trainable
^^^^^^^^^^^^^^^^^^^^^
    | Type: bool
    | Default: ``True``

    Whether the Bessel radial basis should be trainable.

basis
^^^^^
    | Type: type
    | Default: ``<class 'nequip.nn.radial_basis.BesselBasis'>``

    The radial basis to use.

Convolution
***********

Basic
-----

num_layers
^^^^^^^^^^
    | Type: int
    | Default: ``3``

    The number of convolution layers.


feature_irreps_hidden
^^^^^^^^^^^^^^^^^^^^^
    | Type: :ref:`Irreps`
    | Default: n/a

    Specifies the irreps and multiplicities of the hidden features.
    Typically, include irreps with all :math:`\ell` values up to :math:`\ell_{\text{max}}` (see `irreps_edge_sh`_), each with both even and odd parity.
    For example, for ``irreps_edge_sh: 1``, one might provide: ``feature_irreps_hidden: 16x0e + 16x0o + 16x1e + 16x1o``.

Advanced
--------

invariant_layers
^^^^^^^^^^^^^^^^
    | Type: int
    | Default: ``1``

    The number of hidden layers in the radial neural network.

invariant_neurons
^^^^^^^^^^^^^^^^^
    | Type: int
    | Default: ``8``

    The width of the hidden layers of the radial neural network.

resnet
^^^^^^
    | Type: bool
    | Default: ``True``

nonlinearity_type
^^^^^^^^^^^^^^^^^
    | Type: str
    | Default: ``gate``

nonlinearity_scalars
^^^^^^^^^^^^^^^^^^^^
    | Type: dict
    | Default: ``{'e': 'ssp', 'o': 'tanh'}``

nonlinearity_gates
^^^^^^^^^^^^^^^^^^
    | Type: dict
    | Default: ``{'e': 'ssp', 'o': 'abs'}``

use_sc
^^^^^^
    | Type: bool
    | Default: ``True``

Output block
************

Basic
-----

conv_to_output_hidden_irreps_out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | Type: :ref:`Irreps`
    | Default: n/a

    The middle (hidden) irreps of the output block. Should only contain irreps that are contained in the output of the network (``0e`` for potentials).

Advanced
--------











