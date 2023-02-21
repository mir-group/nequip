"""
Config tests
"""
import pytest

from os import remove

from nequip.utils import Config

# set up two config to test
minimal_config = dict(stringv="3x0e", intv=1, nonev=None, boolv=True)
configs = [dict(), minimal_config]
config_testlist = pytest.mark.parametrize("config", configs, indirect=True)
one_test = pytest.mark.parametrize("config", configs[-1:], indirect=True)


@pytest.fixture(scope="class")
def config(request):
    """
    Generate a class instance with minimal configurations
    """
    c = Config(request.param)
    yield c
    del c


class TestConfigSetUp:
    """
    test initialization
    """

    @config_testlist
    def test_init(self, config):
        assert isinstance(config, Config)

    @config_testlist
    def test_set_attr(self, config):

        dict_config = Config.as_dict(config)
        config.intv = 2
        dict_config["intv"] = 2

        assert Config.as_dict(config) == dict_config
        print("dict", Config.as_dict(config))

    @config_testlist
    def test_get_attr(self, config):

        config.intv = 2
        assert config["intv"] == config.intv


class TestConfigIO:
    """
    test I/O methods
    """

    filename = "test_utils_config_save"

    @config_testlist
    def test_repr(self, config):
        s = repr(config)
        print(s)

    @one_test
    def test_save_yaml(self, config):
        config.save(filename=f"{self.filename}.yaml")

    @one_test
    def test_load_yaml(self, config):
        config2 = config.load(filename=f"{self.filename}.yaml")
        assert Config.as_dict(config) == dict(config2)
        remove(f"{self.filename}.yaml")


class TestConfigUpdate:
    """
    test update and type hints
    """

    @config_testlist
    def test_update(self, config):

        dict_config = Config.as_dict(config)
        dict_config["new_intv"] = 9

        newdict = {"new_intv": 9}

        config.update(newdict)

        assert Config.as_dict(config) == dict_config

    @config_testlist
    def test_update_settype(self, config):

        config.set_type("floatv", float)

        assert config.get_type("floatv") == float

        config._floatv2_type = float

        assert config.get_type("floatv2") == float

        config.update({"_floatv3_type": float})

        assert config.get_type("floatv3") == float

    @config_testlist
    def test_update_difftype(self, config):

        config._new_intv_type = int
        config.update({"new_intv": 3.6})
        assert config.new_intv == 3

    @config_testlist
    def test_update_error(self, config):

        with pytest.raises(TypeError) as excinfo:

            config._intv_type = int
            newdict = {"intv": "3x0e"}
            config.update(newdict)

        assert str(excinfo.value).startswith("Wrong Type:")


class TestContainer:
    @one_test
    def test_items(self, config):
        keys = []
        for k, v in config.items():
            keys += [k]
        assert keys == list(config.keys())

    @one_test
    def test_contains(self, config):
        key = list(config.keys())[0]
        assert key in config

    @one_test
    def test_get(self, config):
        config.this = 2
        assert config.get("this") == 2
