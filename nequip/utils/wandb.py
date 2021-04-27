import os
import wandb


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
    # download from wandb set up
    config.update(dict(wandb.config))
    wandb.config.update(dict(run_id=config.run_id), allow_val_change=True)
    return config


def resume(config):
    # resume to the old wandb run
    wandb.init(
        project=config.wandb_project,
        resume="must",
        id=config.run_id,
    )
