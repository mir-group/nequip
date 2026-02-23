# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
"""
Model train-time compilation test mixin.

This mixin provides tests for train-time compilation workflows, i.e.
``compile_mode="compile"`` model construction and execution correctness.
"""

import pytest
import copy

from nequip.utils.test import override_irreps_debug
from nequip.utils.versions import _TORCH_GE_2_6

from .utils import compare_output_and_gradients
from .model_tests_basic import EnergyModelTestsMixin


class TrainTimeCompileMixin(EnergyModelTestsMixin):
    """
    Model train-time compilation tests.

    Inherits from EnergyModelTestsMixin (which inherits from BasicModelTestsMixin).
    These tests verify that train-time compiled models produce outputs and gradients
    consistent with the eager model.
    """

    @pytest.fixture(scope="class")
    def train_time_compile_tol(self, model_dtype):
        """May be overridden by subclasses.

        Returns tolerance for train-time compilation tests based on ``model_dtype``.
        """
        return {"float32": 5e-5, "float64": 1e-12}[model_dtype]

    @pytest.fixture(scope="class", params=[None])
    def train_time_compile_modifiers(self, request):
        """Implemented by subclasses.

        Returns a callable that handles model modification and constraints for
        train-time compile tests.
        The callable signature is: ``(device, model_dtype) -> modifiers_list``

        Args:
            device: target device (``"cpu"`` or ``"cuda"``)
            model_dtype: model dtype string (``"float32"`` or ``"float64"``)

        Returns:
            modifiers_list: list of modifier dicts to apply via ``nequip.model.modify``
                or calls ``pytest.skip()`` to skip the test.

        Default is ``None`` (no modifiers applied).
        """
        return request.param

    @override_irreps_debug(False)
    def test_train_time_compile(
        self,
        model,
        model_test_data,
        device,
        model_dtype,
        train_time_compile_modifiers,
        train_time_compile_tol,
    ):
        """
        Test train-time compilation correctness, i.e.
        ``make_fx`` -> ``export`` -> ``AOTAutograd`` behavior.

        This test builds a model with ``compile_mode="compile"``, optionally applies
        modifiers, and compares outputs and parameter gradients against the eager model.
        """
        if not _TORCH_GE_2_6:
            pytest.skip("PT2 compile tests skipped for torch < 2.6")

        # TODO: have better way to test since problem might be version dependent now
        # see https://github.com/pytorch/pytorch/issues/146390
        # i.e. some ops have problems in torch 2.6, but may be ok in torch 2.7
        if device == "cpu":
            pytest.skip(
                "compile tests are skipped for CPU as there are known compilation bugs for both NequIP and Allegro models on CPU"
            )

        instance, config, _ = model

        modifier_configs = []
        if train_time_compile_modifiers is not None:
            modifier_configs = train_time_compile_modifiers(device, model_dtype)

        config = copy.deepcopy(config)
        config["compile_mode"] = "compile"

        if modifier_configs:
            config = {
                "_target_": "nequip.model.modify",
                "modifiers": modifier_configs,
                "model": config,
            }

        compile_model = self.make_model(config, device=device)

        compare_output_and_gradients(
            modelA=instance,
            modelB=compile_model,
            model_test_data=model_test_data,
            tol=train_time_compile_tol,
        )
