Overview
========

Introduction
############
``nequip`` is a framework for E(3)-equivariant machine learning on atomic data, with a particular focus on machine learning interatomic potentials.

Note that ``nequip`` is a framework that includes, but is not limited to, the NequIP model architecture. In particular, it is also used by the Allegro model architecutre.

Citing NequIP
#############
If you use ``nequip`` in your research, please cite our `article <https://doi.org/10.1038/s41467-022-29939-5>`_:

.. code-block:: bibtex

    @article{batzner_e3-equivariant_2022,
      title = {E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials},
      author = {Batzner, Simon and Musaelian, Albert and Sun, Lixin and Geiger, Mario and Mailoa, Jonathan P. and Kornbluth, Mordechai and Molinari, Nicola and Smidt, Tess E. and Kozinsky, Boris},
      year = {2022},
      month = may,
      journal = {Nature Communications},
      volume = {13},
      number = {1},
      pages = {2453},
      issn = {2041-1723},
      doi = {10.1038/s41467-022-29939-5},
    }

The theory behind NequIP is described in our `article <https://doi.org/10.1038/s41467-022-29939-5>`_ above.
NequIP's backend builds on `e3nn <https://e3nn.org>`_, a general framework for building E(3)-equivariant
neural networks (1). If you use this repository in your work, please consider citing ``nequip`` and ``e3nn`` (2):

 1. https://e3nn.org
 2. https://doi.org/10.5281/zenodo.3724963

