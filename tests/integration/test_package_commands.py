# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.
import subprocess
import tempfile
import zipfile

import pytest

from conftest import _check_and_print


@pytest.fixture(scope="session")
def package_from_training(fake_model_training_session, tmp_path_factory):
    config, tmpdir, env, model_dtype = fake_model_training_session
    pkg_path = tmp_path_factory.mktemp("pkg_cmd") / "model.nequip.zip"
    retcode = subprocess.run(
        ["nequip-package", "build", f"{tmpdir}/best.ckpt", str(pkg_path)],
        cwd=tmpdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    assert pkg_path.is_file(), "`nequip-package build` did not create file"
    return pkg_path, env


def test_list_and_diff(package_from_training):
    pkg_path, env = package_from_training

    # === test list ===
    retcode = subprocess.run(
        ["nequip-package", "list", str(pkg_path)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    output = retcode.stdout.decode()
    assert ".storage" not in output

    # each line has format "{size:>12}  {zip_path}"; collect all paths
    zip_paths = {}
    for line in output.splitlines():
        parts = line.split()
        assert parts[0].isdigit(), f"expected size field, got: {line!r}"
        zip_paths[parts[1]] = True
    for resource_suffix in [
        "model/config.yaml",
        "model/package_metadata.txt",
        "model/example_data.pkl",
    ]:
        assert any(p.endswith(resource_suffix) for p in zip_paths), (
            f"{resource_suffix} not found in list output"
        )

    # === test diff using the path found via list ===
    # users are expected to copy the path from `list` output and pass it to `diff`
    config_zip_path = next(p for p in zip_paths if p.endswith("model/config.yaml"))
    with zipfile.ZipFile(pkg_path) as zf:
        data = zf.read(config_zip_path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as f:
        f.write(data)
        local_path = f.name
    retcode = subprocess.run(
        ["nequip-package", "diff", str(pkg_path), config_zip_path, local_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    assert "(no differences)" in retcode.stdout.decode()


def test_modify(package_from_training, tmp_path):
    pkg_path, env = package_from_training
    pkg_mtime = pkg_path.stat().st_mtime
    out_path = tmp_path / "modified.nequip.zip"
    retcode = subprocess.run(
        [
            "nequip-package",
            "modify",
            str(pkg_path),
            str(out_path),
            "--modifier",
            "disable_ForceStressOutput",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _check_and_print(retcode)
    assert out_path.exists()
    with zipfile.ZipFile(out_path) as zf:
        names = zf.namelist()
    assert any("model.pkl" in n for n in names)
    # original package must be untouched
    assert pkg_path.stat().st_mtime == pkg_mtime


def test_modify_same_path_error(package_from_training):
    pkg_path, env = package_from_training
    retcode = subprocess.run(
        [
            "nequip-package",
            "modify",
            str(pkg_path),
            str(pkg_path),
            "--modifier",
            "disable_ForceStressOutput",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert retcode.returncode != 0


def test_modify_unknown_modifier_error(package_from_training, tmp_path):
    pkg_path, env = package_from_training
    out_path = tmp_path / "out.nequip.zip"
    retcode = subprocess.run(
        [
            "nequip-package",
            "modify",
            str(pkg_path),
            str(out_path),
            "--modifier",
            "nonexistent_modifier_xyz",
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert retcode.returncode != 0
