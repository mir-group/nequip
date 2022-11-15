from nequip.utils.unittests import CONFTEST_PATH

# like `source` in bash
with open(CONFTEST_PATH) as f:
    exec(f.read())
