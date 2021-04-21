import os
import wandb


def init_n_update(config):
    # download from wandb set up
    config.run_id = wandb.util.generate_id()
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = config.run_id

    wandb.init(project=config.wandb_project, config=dict(config))
    # download from wandb set up
    config.update(dict(wandb.config))
    config.run_id = wandb.run.id
    wandb.run.name = config.run_name
    wandb.config.update(dict(config), allow_val_change=True)
    return config


def resume(config):
    # store this id to use it later when resuming
    os.environ["WANDB_RESUME"] = "must"
    wandb.init(project=config.wandb_project, config=dict(config), resume=config.run_id)
