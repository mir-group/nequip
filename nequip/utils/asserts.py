# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.


def assert_package_extension(path, what: str = "packaged model file") -> None:
    assert str(path).endswith(".nequip.zip"), (
        f"The {what} should have the extension `.nequip.zip`, but found {str(path)}"
    )
