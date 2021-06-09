import os
import wandb
import logging
from wandb.util import json_friendly_val


def init_n_update(config):
    # download from wandb set up
    config.run_id = wandb.util.generate_id()

    wandb.init(
        project=config.wandb_project,
        config=dict(config),
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
