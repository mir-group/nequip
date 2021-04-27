""" Start or automatically restart training.

Arguments: config.yaml

config.yaml: requeue=True, and workdir, root, run_name have to be unique.
"""
from os.path import isfile

from .train import fresh_start, parse_command_line
from .restart import restart


def main(args=None):
    config = parse_command_line()
    requeue(config)


def requeue(config):

    assert config.get(
        "requeue", False
    ), "This script only works for configs with `requeue` explicitly set to True. Be careful!!"
    for key in ["workdir", "root", "run_name"]:
        assert isinstance(
            config[key], str
        ), f"{key} has to be defined for requeue script"

    found_restart_file = isfile(config.workdir + "/trainer.pth")
    config.restart = found_restart_file
    config.append = found_restart_file
    config.force_append = True

    # for fresh new train
    if not found_restart_file:
        config.run_time = 1
        fresh_start(config)
    else:
        restart(config.workdir + "/trainer.pth", config, mode="requeue")

    return


if __name__ == "__main__":
    main()
