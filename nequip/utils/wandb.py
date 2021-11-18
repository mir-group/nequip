import wandb
import logging
from wandb.util import json_friendly_val


def init_n_update(config):
    conf_dict = dict(config)
    # wandb mangles keys (in terms of type) as well, but we can't easily correct that because there are many ambiguous edge cases. (E.g. string "-1" vs int -1 as keys, are they different config keys?)
    if any(not isinstance(k, str) for k in conf_dict.keys()):
        raise TypeError(
            "Due to wandb limitations, only string keys are supported in configurations."
        )

    # download from wandb set up
    config.run_id = wandb.util.generate_id()

    wandb.init(
        project=config.wandb_project,
        config=conf_dict,
        name=config.run_name,
        resume="allow",
        id=config.run_id,
    )
    # # download from wandb set up
    updated_parameters = dict(wandb.config)
    for k, v_new in updated_parameters.items():
        skip = False
        if k in config.keys():
            # double check the one sanitized by wandb
            v_old = json_friendly_val(config[k])
            if repr(v_new) == repr(v_old):
                skip = True
        if skip:
            logging.info(f"# skipping wandb update {k} from {v_old} to {v_new}")
        else:
            config.update({k: v_new})
            logging.info(f"# wandb update {k} from {v_old} to {v_new}")
    return config


def resume(config):
    # resume to the old wandb run
    wandb.init(
        project=config.wandb_project,
        resume="must",
        id=config.run_id,
    )
