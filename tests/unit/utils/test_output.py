"""
Config tests
"""
import pytest
import tempfile

from os.path import isdir

from nequip.utils.output import Output

# set up two config to test
minimal_config = dict(stringv="3x0e", intv=1, nonev=None, boolv=True)
configs_to_test = [dict(), minimal_config]


class TestInit:
    def test_empty_init(self, root):
        output = Output(root=root, run_name="test")
        print(output.root)
        print(output.workdir)
        assert isdir(output.root)
        assert isdir(output.workdir)


class TestProject:
    def test_empty_init(self, root):
        output = Output(root=root, run_name="not_default")
        assert isdir(output.root)
        assert isdir(output.workdir)
        assert "not_default" in output.workdir


class TestReload:
    @pytest.mark.parametrize("append", [True, False])
    def test_restart(self, append):
        pass


@pytest.fixture(scope="class")
def root():
    with tempfile.TemporaryDirectory(prefix="output") as path:
        yield path
