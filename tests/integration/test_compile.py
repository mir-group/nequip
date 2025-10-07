import pytest
import torch
from pathlib import Path
from nequip.scripts.compile import main


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
@pytest.mark.parametrize(
    "mode,extension", [("aotinductor", ".nequip.pt2"), ("torchscript", ".nequip.pth")]
)
def test_compile_modes(tmp_path_factory, device, mode, extension):
    tmp_path = tmp_path_factory.mktemp("nequip_compiled")
    output_model_name = f"mir-group__NequIP-OAM-L__0.1{extension}"
    output_path = Path(tmp_path) / output_model_name

    args = [
        "nequip.net:mir-group/NequIP-OAM-L:0.1",
        str(output_path),
        "--mode",
        mode,
        "--device",
        device,
    ]

    if mode == "aotinductor":
        args.extend(["--target", "ase"])

    main(args=args)

    assert output_path.exists()
    assert output_path.is_file()
