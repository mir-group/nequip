General
=======

Basic
-----

root
^^^^
    | Type:
    | Default: n/a

run_name
^^^^^^^^
    | Type: path
    | Default: n/a

    ``run_name`` specifies something about whatever

Advanced
--------

allow_tf32
^^^^^^^^^^
    | Type: bool
    | Default: ``False``

    If ``False``, the use of NVIDIA's TensorFloat32 on Tensor Cores (Ampere architecture and later) will be disabled.
    If ``True``, the PyTorch defaults (use anywhere possible) will remain.