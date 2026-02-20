Inference
=========

Extension packages can perform inference using NequIP framework models through existing integrations or by extending them with custom functionality.

Integrations
------------

The NequIP framework provides pre-built integrations that extension packages can leverage:

ASE Integration
~~~~~~~~~~~~~~~

The :class:`~nequip.integrations.ase.NequIPCalculator` provides ASE calculator functionality. Extension packages can:

- Subclass :class:`~nequip.integrations.ase.NequIPCalculator` to add custom behavior
- Override the :meth:`~nequip.integrations.ase.NequIPCalculator.save_extra_outputs` method to process additional model outputs
- Extend the supported properties by modifying :attr:`~nequip.integrations.ase.NequIPCalculator.implemented_properties`

See the `ASE integration documentation <../../integrations/ase.html>`__ for usage details.

LAMMPS ML-IAP Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~nequip.integrations.lammps_mliap.lmp_mliap_wrapper.NequIPLAMMPSMLIAPWrapper` is the LAMMPS ML-IAP integration module. Extension packages that implement basic interatomic potential models in a compliant manner can be used with this wrapper.

Custom Integrations
~~~~~~~~~~~~~~~~~~~

For other simulation packages or custom workflows, extension packages can build upon the underlying model loading and inference utilities. The NequIP framework provides two main model loading functions for inference:

.. autofunction:: nequip.model.saved_models.load_utils.load_saved_model

Use this function for loading trained models from:

- `Checkpoint files <../../guide/getting-started/files.html#checkpoint-files>`__ (``.ckpt``) saved during training
- `Package files <../../guide/getting-started/files.html#package-files>`__ (``.nequip.zip``) created with ``nequip-package``

.. autofunction:: nequip.model.inference_models.load_compiled_model

Use this function for loading compiled models created with `nequip-compile <../../guide/getting-started/workflow.html#compilation>`__, including:

- TorchScript models (``.nequip.ts``)
- AOT Inductor compiled models (``.nequip.pt2``)

AOT Inductor Compilation Targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NequIP framework provides predefined compilation targets for common use cases:

- **ase**: Optimized for ASE calculator integration with standard ASE outputs
- **batch**: Supports batched inference with additional batch dimension handling

Extension packages can register custom AOT Inductor input/output specifications using the following function:

.. autofunction:: nequip.scripts._compile_utils.register_compile_targets

Contact
-------

For questions about developing inference capabilities in extension packages:

- Open an issue or start a discussion on the `NequIP GitHub <https://github.com/mir-group/nequip>`__
- Join the NequIP community on `Zulip <https://forms.gle/mEuonVCHdsgTtLXy7>`__ for developer-focused discussions
- Email us at nequip@g.harvard.edu
