""" Nequip.train.trainer

Todo:

isolate the loss function from the training procedure
enable wandb resume
make an interface with ray

"""

import inspect
import logging
import torch
import yaml
import numpy as np

from copy import deepcopy
from os.path import isfile
from time import perf_counter
from typing import Optional, Union

from nequip.data import DataLoader, AtomicData, AtomicDataDict
from nequip.utils import (
    Output,
    instantiate_from_cls_name,
    instantiate,
    save_file,
    load_file,
)

from .loss import Loss
from ._key import *


class Trainer:
    """Class to train a model to minimize forces

    This class isolate the logging and callback functions from the training procedure.
    The class instance can be easily save/load for restart;
    There are options to append the old logging file or to restart a new file,
    and options to append old model files or to save in a new path.

    Examples:

    To start a training.

    '''python
    trainer = Trainer(model, learning_rate=1e-2)
    trainer.set_dataset(dataset)
    trainer.train()
    '''

    To load and restart a training.

    '''python
    trainer = Trainer.from_file(filename)
    trainer.set_dataset(dataset)
    trainer.train()
    '''

    To load, slightly revise content and then resume training
    '''python
    dictionary = nequip.utils.savenload.load_file(
            supported_formats=dict(torch=["pt", "pth"]),
            filename=filename,
            enforced_format="torch",
        )
    dictionary["progress"]["stop_arg"] = None
    dictionary["max_epochs"] += 100
    trainer = Trainer.from_dict(
        dictionary, append=False
    )
    trainer.train()
    '''

    For a fresh run, the simulation results will be stored in the 'root/project' folder. With
    - "log" : plain text information about the whole training process
    - "metrics_epoch.txt" : txt matrice format (readable by np.loadtxt) with loss/mae from training and validation of each epoch
    - "metrics_batch.txt" : txt matrice format (readable by np.loadtxt) with training loss/mae of each batch
    - "best_model.pth": the best model. save the model when validation mae goes lower
    - "last_model.pth": the last model. save the model every log_epoch_freq epoch
    - "trainer_save.pth": all the training information. The file used for loading and restart

    For restart run, the default set up is to not append to the original folders and files.
    The Output class will automatically build a folder call root/project_
    If append mode is on, the log file will be appended and the best model and last model will be overwritten.

    More examples can be found in tests/train/test_trainer.py

    Note:
        The data in the dataloader has to be DataLoader and Data in torch_geometric.data

        Only optimizers from torch.optim and schedulers from torch.optim.lr_scheduler
        are supported

        Extra arguments needed by optimizer and scheduler can be directly taken in as
        long as they are exactly the same as the ones listed in the init functions. To
        avoid ambituity, one can also address the argument with "lr_" and "optim_" prefixes
        for the lr_scheduler and optimizer, correspondingly.
        Or save them in a dictionary as "optim_params" and "lr_scheduler_params"

        The priority of parameters for optimizers/schedulers is:

            key < optim_key < optim_params[key]

    For developer,

    - all parameters added to the init function arguments will be automatically
      saved as class attribute with the same name, and saved to the pth file
    - To store and recover new additional attributes, it needs explicit statement in the as_dict and from_dict
    - To check which parameters were actually used for optimizer, scheduler and loaders initialization, please use verbose=debug.
      The screen output will list all the details


    Args:
        model: neural network model

        seed (int): random see number

        project (str): project name.
        root (str): the name of root dir to make work folders
        timestr (optional, str): unique string to differentiate this trainer from others.

        restart (bool) : If true, the init_model function will not be callsed. Default: False
        append (bool): If true, the preexisted workfolder and files will be overwritten. And log files will be appended

        loss_coeffs (dict): dictionary to store coefficient and loss functions
        atomic_weight_on (dict): if true, the weights in dataset will be used for loss/mae calculations.

        max_epochs (int): maximum number of epochs

        lr_sched (optional): scheduler
        learning_rate (float): initial learning rate
        lr_scheduler_name (str): scheduler name
        lr_scheduler_params (dict): parameters to initialize the scheduler

        optim (): optimizer
        optimizer_name (str): name for optimizer
        optim_params (dict): parameters to initialize the optimizer

        batch_size (int): size of each batch
        shuffle (bool): parameters for dataloader
        n_train (int): # of frames for training
        n_val (int): # of frames for validation
        exclude_keys (list):  parameters for dataloader
        train_idcs (optional, list):  list of frames to use for training
        val_idcs (list):  list of frames to use for validation
        train_val_split (str):  "random" or "sequential"
        loader_params (dict):  Parameters for dataloader

        init_callbacks (list): list of callback function at the begining of the training
        end_of_epoch_callbacks (list): list of callback functions at the end of each epoch
        end_of_batch_callbacks (list): list of callback functions at the end of each batch
        end_of_train_callbacks (list): list of callback functions between traing/validation
        final_callbacks (list): list of callback functions at the end of the training

        log_batch_freq (int): frequency to log at the end of a batch
        log_epoch_freq (int): frequency to save at the end of an epoch

        verbose (str): verbosity level, i.e. "INFO", "WARNING", "DEBUG". case insensitive

    Additional Attributes:

        init_params (list): list of parameters needed to reconstruct this instance
        device : torch device
        optim: optimizer
        lr_sched: scheduler
        dl_train (DataLoader): training data
        dl_val (DataLoader): test data
        iepoch (int): # of epoches ran
        stop_arg (str): reason why the training stops
        batch_mae (float): the mae of the latest batch
        mae_dict (dict): all loss, mae of the latest validation
        best_val_metrics (float): current best validation mae
        best_epoch (float): current best epoch
        best_model_path (str): path to save the best model
        last_model_path (str): path to save the latest model
        trainer_save_path (str): path to save the trainer.
             Default is trainer.(date).pth at the current folder


    The pseudocode of the workflow and location of the callback functions

    ```
    init():
        initialize optimizer, schduler and loss function

    train():
       init model
       init_callbacks
       while (not stop):
            training batches
                end_of_batch_callbacks
            end_of_train_callbacks
            validation_batches
            end_of_epoch_callbacks
       final_callbacks
    ```
    """

    def __init__(
        self,
        model,
        project: Optional[str] = None,
        root: Optional[str] = None,
        timestr: Optional[str] = None,
        seed: Optional[int] = None,
        restart: bool = False,
        append: bool = False,
        loss_coeffs: Union[dict, str] = AtomicDataDict.TOTAL_ENERGY_KEY,
        metrics_key: str = ABBREV.get(LOSS_KEY, LOSS_KEY),
        atomic_weight_on: bool = False,
        max_epochs: int = 1000000,
        lr_sched=None,
        learning_rate: float = 1e-2,
        lr_scheduler_name: str = "ReduceLROnPlateau",
        lr_scheduler_params: Optional[dict] = None,
        optim=None,
        optimizer_name: str = "Adam",
        optim_params: Optional[dict] = None,
        exclude_keys: list = [],
        batch_size: int = 5,
        shuffle: bool = True,
        n_train: Optional[int] = None,
        n_val: Optional[int] = None,
        train_idcs: Optional[list] = None,
        val_idcs: Optional[list] = None,
        train_val_split: str = "random",
        loader_params: Optional[dict] = None,
        init_callbacks: list = [],
        end_of_epoch_callbacks: list = [],
        end_of_batch_callbacks: list = [],
        end_of_train_callbacks: list = [],
        final_callbacks: list = [],
        log_batch_freq: int = 1,
        log_epoch_freq: int = 1,
        verbose="INFO",
        **kwargs,
    ):
        self._initialized = False
        logging.debug("* Initialize Trainer")

        # store all init arguments
        self.root = root
        self.model = model
        self.optim = optim
        self.lr_sched = lr_sched

        for key in self.init_params:
            setattr(self, key, locals()[key])

        output = Output.get_output(timestr, self)

        for key, value in output.as_dict().items():
            setattr(self, key, value)
        if self.logfile is None:
            self.logfile = output.open_logfile("log", propagate=True)
        self.epoch_log = output.open_logfile("metrics_epoch.txt", propagate=False)
        self.batch_log = {
            TRAIN: output.open_logfile("metrics_batch_train.txt", propagate=False),
            VALIDATION: output.open_logfile("metrics_batch_val.txt", propagate=False),
        }

        # add filenames if not defined
        self.best_model_path = output.generate_file("best_model.pth")
        self.last_model_path = output.generate_file("last_model.pth")
        self.trainer_save_path = output.generate_file("trainer.pth")

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Torch device: {self.device}")

        # sort out all the other parameters
        # for samplers, optimizer and scheduler
        self.kwargs = deepcopy(kwargs)
        self.optim_params = deepcopy(optim_params)
        self.lr_scheduler_params = deepcopy(lr_scheduler_params)
        self.loader_params = deepcopy(loader_params)

        # initialize the optimizer and scheduler, the params will be updated in the function
        self.init()

        self.statistics = {}

        if not (restart and append):
            self.log_dictionary(self.as_dict(), name="Initialization")

        logging.debug("! Done Initialize Trainer")

    @property
    def init_params(self):
        d = inspect.signature(Trainer.__init__)
        names = list(d.parameters.keys())
        for key in [
            "model",
            "optim",
            "lr_sched",
            "self",
            "kwargs",
        ]:
            if key in names:
                names.remove(key)
        return names

    @property
    def logger(self):
        return logging.getLogger(self.logfile)

    @property
    def epoch_logger(self):
        return logging.getLogger(self.epoch_log)

    def as_dict(self, state_dict: bool = False, training_progress: bool = False):
        """convert instance to a dictionary
        Args:

        state_dict (bool): if True, the weights and bias will also be stored.
              When best_model_path and last_model_path are not defined,
              the weights of the model will be explicitly stored in the dictionary
        """

        dictionary = {}
        # collect all init arguments
        # note that kwargs will not be stored if those values are not in
        # load_params, optim_params and lr_schduler_params
        for key in self.init_params:
            dictionary[key] = getattr(self, key, None)
        dictionary["kwargs"] = getattr(self, "kwargs", {})

        if state_dict:
            dictionary["state_dict"] = {}
            dictionary["state_dict"]["optim"] = self.optim.state_dict()
            dictionary["state_dict"]["lr_sched"] = self.lr_sched.state_dict()

        if hasattr(self.model, "save") and not issubclass(
            type(self.model), torch.jit.ScriptModule
        ):
            dictionary["model_class"] = type(self.model)

        if training_progress:
            dictionary["progress"] = {}
            for key in ["iepoch", "best_epoch"]:
                dictionary["progress"][key] = self.__dict__.get(key, -1)
            dictionary["progress"]["best_val_metrics"] = self.__dict__.get(
                "best_val_metrics", float("inf")
            )
            dictionary["progress"]["stop_arg"] = self.__dict__.get("stop_arg", None)
            dictionary["progress"]["best_model_path"] = self.best_model_path
            dictionary["progress"]["last_model_path"] = self.last_model_path
            dictionary["progress"]["trainer_save_path"] = self.trainer_save_path

        return dictionary

    def save(self, filename, format=None):
        """save the file as filename

        Args:

        filename (str): name of the file
        format (str): format of the file. yaml and json format will not save the weights.
        """

        logger = self.logger

        state_dict = (
            True
            if format == "torch"
            or filename.endswith(".pth")
            or filename.endswith(".pt")
            else False
        )

        filename = save_file(
            item=self.as_dict(state_dict=state_dict, training_progress=True),
            supported_formats=dict(torch=["pth", "pt"], yaml=["yaml"], json=["json"]),
            filename=filename,
            enforced_format=format,
        )
        logger.debug(f"Saving trainer to {filename}")

        if hasattr(self.model, "save"):
            self.model.save(self.last_model_path)
        else:
            torch.save(self.model, self.last_model_path)
        logger.debug(f"Saving last model to to {self.last_model_path}")

        return filename

    @staticmethod
    def from_file(
        filename: str, format: Optional[str] = None, append: Optional[bool] = None
    ):
        """load a model from file

        Args:

        filename (str): name of the file
        append (bool): if True, append the old model files and append the same logfile
        """

        dictionary = load_file(
            supported_formats=dict(torch=["pth", "pt"], yaml=["yaml"], json=["json"]),
            filename=filename,
            enforced_format=format,
        )
        return Trainer.from_dict(dictionary, append)

    @staticmethod
    def from_dict(dictionary, append: Optional[bool] = None):
        """load model from dictionary

        Args:

        dictionary (dict):
        append (bool): if True, append the old model files and append the same logfile
        """

        d = deepcopy(dictionary)
        kwargs = d.pop("kwargs", {})
        d.update(kwargs)

        # update the restart and append option
        d["restart"] = True
        if append is not None:
            d["append"] = append

        # update the file and folder name
        output = Output.from_config(d)
        d.update(output.as_dict())

        model = None
        iepoch = 0
        if "progress" in d:
            progress = d["progress"]
            stop_arg = progress["stop_arg"]
            if stop_arg is not None:
                raise RuntimeError(
                    f"The previous run has properly stopped with {stop_arg}."
                    "Please either increase the max_epoch or change early stop criteria"
                )

            # load the model from file
            iepoch = progress["iepoch"]
            if isfile(progress["last_model_path"]):
                load_path = progress["last_model_path"]
                iepoch = progress["iepoch"]
            elif isfile(progress["best_model_path"]):
                load_path = progress["best_model_path"]
                iepoch = progress["best_epoch"]
            else:
                raise AttributeError("model weights & bias are not saved")

            if "model_class" in d:
                model = d["model_class"].load(load_path)
            else:
                model = torch.load(load_path)
            logging.debug(f"Reload the model from {load_path}")

            d.pop("progress")

        state_dict = d.pop("state_dict", None)

        trainer = Trainer(model=model, **d)

        if state_dict is not None and trainer.model is not None:
            trainer.optim.load_state_dict(state_dict["optim"])
            trainer.lr_sched.load_state_dict(state_dict["lr_sched"])
            logging.debug("Reload optimizer and scheduler states")

        if "progress" in d:
            trainer.best_val_metrics = progress["best_val_metrics"]
            trainer.best_epoch = progress["best_epoch"]
        else:
            trainer.best_val_metrics = float("inf")
            trainer.best_epoch = 0
        trainer.iepoch = iepoch

        return trainer

    def init(self):
        """ initialize optimizer """

        if self.model is None:
            return

        if self.optim is None:
            self.optim, self.optim_params = instantiate_from_cls_name(
                module=torch.optim,
                class_name=self.optimizer_name,
                prefix="optim",
                positional_args=dict(
                    params=self.model.parameters(), lr=self.learning_rate
                ),
                all_args=self.kwargs,
                optional_args=self.optim_params,
            )

        if self.lr_sched is None:
            assert (
                self.lr_scheduler_name
                in ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau"]
            ) or (
                (len(self.end_of_epoch_callbacks) + len(self.end_of_batch_callbacks))
                > 0
            ), f"{self.lr_scheduler_name} cannot be used unless callback functions are defined"
            self.lr_sched, self.lr_scheduler_params = instantiate_from_cls_name(
                module=torch.optim.lr_scheduler,
                class_name=self.lr_scheduler_name,
                prefix="lr",
                positional_args=dict(optimizer=self.optim),
                optional_args=self.lr_scheduler_params,
                all_args=self.kwargs,
            )

        self.loss = Loss(self.loss_coeffs, atomic_weight_on=self.atomic_weight_on)
        self._initialized = True

    def init_model(self):

        logger = self.logger
        logger.info(
            "Number of weights: {}".format(
                sum(p.numel() for p in self.model.parameters())
            )
        )

    def train(self):
        """Training"""
        if getattr(self, "dl_train", None) is None:
            raise RuntimeError("You must call `set_dataset()` before calling `train()`")
        if not self._initialized:
            self.init()

        self.model.to(self.device)

        if not self.restart:
            self.init_model()

        for callback in self.init_callbacks:
            callback(self)

        self.init_log()
        self.wall = perf_counter()

        stop = False
        if not self.restart:
            self.best_val_metrics = float("inf")
            self.best_epoch = 0
            self.iepoch = 0

        while self.iepoch < self.max_epochs and not stop:

            early_stop = self.epoch_step()
            if early_stop:
                stop = False
                self.stop_arg = "early stop"

            self.iepoch += 1

        if not stop:
            self.stop_arg = "max epochs"

        for callback in self.final_callbacks:
            callback(self)

        self.final_log()

        self.save(self.trainer_save_path)

    def batch_step(self, data, n_batches, validation=False):

        self.model.train()

        # Do any target rescaling
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)
        if hasattr(self.model, "unscale"):
            # This means that self.model is RescaleOutputs
            # this will normalize the targets
            # in validation (eval mode), it does nothing
            # in train mode, if normalizes the targets
            data = self.model.unscale(data)

        # Run model
        out = self.model(data)

        # If we're in evaluation mode (i.e. validation), then
        # data's target prop is unnormalized, and out's has been rescaled to be in the same units
        # If we're in training, data's target prop has been normalized, and out's hasn't been touched, so they're both in normalized units
        # Note that either way all normalization was handled internally by RescaleOutput

        loss, loss_contrib = self.loss(pred=out, ref=data)

        if not validation:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if self.lr_scheduler_name == "CosineAnnealingWarmRestarts":
                self.lr_sched.step(self.iepoch + self.ibatch / n_batches)

        # save loss stats
        with torch.no_grad():
            mae, mae_contrib = self.loss.mae(pred=out, ref=data)
            scaled_loss_contrib = {}
            if hasattr(self.model, "scale"):

                for key in mae_contrib:
                    mae_contrib[key] = self.model.scale(
                        mae_contrib[key], force_process=True, do_shift=False
                    )

                # TO DO, this evetually needs to be removed. no guarantee that a loss is MSE
                for key in loss_contrib:

                    scaled_loss_contrib[key] = {
                        k: torch.clone(v) for k, v in loss_contrib[key].items()
                    }

                    scaled_loss_contrib[key] = self.model.scale(
                        scaled_loss_contrib[key],
                        force_process=True,
                        do_shift=False,
                        do_scale=True,
                    )

                    keys = [k for k in scaled_loss_contrib[key] if k in self.loss.funcs]
                    keys = [
                        k
                        for k in keys
                        if "mse" in type(self.loss.funcs[k].func).__name__.lower()
                    ]
                    if len(keys) > 0:
                        scaled_loss_contrib[key] = self.model.scale(
                            scaled_loss_contrib[key],
                            force_process=True,
                            do_shift=False,
                            do_scale=True,
                        )

            self.batch_loss = loss.detach()
            self.batch_scaled_loss_contrib = {
                k1: {k2: v2.detach() for k2, v2 in v1.items()}
                for k1, v1 in scaled_loss_contrib.items()
            }
            self.batch_loss_contrib = {
                k1: {k2: v2.detach() for k2, v2 in v1.items()}
                for k1, v1 in loss_contrib.items()
            }
            self.batch_mae = mae.detach()
            self.batch_mae_contrib = {
                k1: {k2: v2.detach() for k2, v2 in v1.items()}
                for k1, v1 in mae_contrib.items()
            }

            self.end_of_batch_log(validation)
            for callback in self.end_of_batch_callbacks:
                callback(self)

    @property
    def early_stop_cond(self):
        """ kill the training early """

        return False

    def epoch_step(self):

        for self.ibatch, batch in enumerate(self.dl_train):
            self.batch_step(
                data=batch,
                n_batches=self.n_train_batches,
                validation=False,
            )

        for callback in self.end_of_train_callbacks:
            callback(self)

        for self.ibatch, batch in enumerate(self.dl_val):
            self.batch_step(data=batch, n_batches=self.n_val_batches, validation=True)

        self.end_of_epoch_log()
        self.end_of_epoch_save()

        if self.lr_scheduler_name == "ReduceLROnPlateau":
            self.lr_sched.step(
                metrics=self.mae_dict[f"{VALIDATION}_{self.metrics_key}"]
            )

        for callback in self.end_of_epoch_callbacks:
            callback(self)

        return self.early_stop_cond

    def log_dictionary(self, dictionary: dict, name: str = ""):
        """
        dump the keys and values of a dictionary
        """

        logger = self.logger
        logger.info(f"* {name}")
        logger.info(yaml.dump(dictionary))

    def end_of_batch_log(self, validation: bool):
        """
        store all the loss/mae of each batch
        """

        category = VALIDATION if validation else TRAIN

        if self.ibatch == 0:
            # initialization
            stat = {
                RMSE_LOSS_KEY: {VALUE_KEY: [], CONTRIB: {}},
                MAE_KEY: {VALUE_KEY: [], CONTRIB: {}},
                LOSS_KEY: {VALUE_KEY: [], CONTRIB: {}},
            }

            for key, d in self.batch_mae_contrib.items():
                for attr, value in d.items():
                    if key != attr:
                        store_key = f"{key}_{ABBREV.get(attr, attr)}"
                    else:
                        store_key = f"{key}"
                    for name, d_cat in stat.items():
                        d_cat[CONTRIB][store_key] = []

            self.statistics[category] = stat
        else:
            stat = self.statistics[category]

        mat_str = f"  {self.iepoch+1:5d} {self.ibatch+1:5d}"
        log_str = f"{mat_str}"
        batch_type = "Validation" if validation else "Training"
        header = f"\n# Epoch\n# batch"
        log_header = f"\n{batch_type}\n# Epoch batch"
        for combo in [
            (RMSE_LOSS_KEY, None, self.batch_scaled_loss_contrib),
            (MAE_KEY, self.batch_mae, self.batch_mae_contrib),
            (LOSS_KEY, self.batch_loss, self.batch_loss_contrib),
        ]:
            name, value, contrib = combo
            short_name = ABBREV.get(name, name)

            if name == LOSS_KEY:

                stat[name][VALUE_KEY] += [value]
                mat_str += f" {value:8.3f}"
                header += f"\n# {name}"
                log_str += f" {value:16.3g}"
                log_header += f" {short_name:>16s}"

            for key, d in contrib.items():

                for attr, value in d.items():

                    print_key = f"{ABBREV.get(key, key)}"
                    store_key = f"{key}"
                    if key != attr:
                        store_key += f"_{ABBREV.get(attr, attr)}"
                        print_key += f"_{ABBREV.get(attr, attr)}"

                    if "mse" in type(self.loss.funcs[attr].func).__name__.lower() and (
                        name == RMSE_LOSS_KEY
                    ):
                        value = torch.sqrt(value)
                    value = value.item()
                    stat[name][CONTRIB][store_key] += [value]

                    mat_str += f" {value:8.3f}"
                    item_name = f"{print_key}_{short_name}"
                    header += f"\n# {item_name}"

                    if name != LOSS_KEY:
                        log_str += f" {value:16.3f}"
                        log_header += f" {item_name:>16s}"

        batch_logger = logging.getLogger(self.batch_log[category])
        if not self.batch_header_print[category]:
            batch_logger.info(header)
            self.batch_header_print[category] = True

        if self.ibatch == 0:
            self.logger.info(log_header)

        batch_logger.info(mat_str)
        if (self.ibatch + 1) % self.log_batch_freq == 0 or (
            self.ibatch + 1
        ) == self.n_train_batches:
            self.logger.info(log_str)

    def end_of_epoch_save(self):
        """
        save model and trainer details
        """
        val_metrics = self.mae_dict[f"{VALIDATION}_{self.metrics_key}"]
        if val_metrics < self.best_val_metrics:
            self.best_val_metrics = val_metrics
            self.best_epoch = self.iepoch
            if hasattr(self.model, "save"):
                self.model.save(self.best_model_path)
            else:
                torch.save(self.model, self.best_model_path)
            self.logger.info(
                f"! Best model {self.best_epoch+1:8d} {self.best_val_metrics:8.3f}"
            )

        if (self.iepoch + 1) % self.log_epoch_freq == 0:
            self.save(self.trainer_save_path)

    def init_log(self):

        self.logger.info("! Starting training ...")
        self.epoch_header_print = False
        self.batch_header_print = {TRAIN: False, VALIDATION: False}

    def final_log(self):

        self.logger.info(f"! Stop training for eaching {self.stop_arg}")
        wall = perf_counter() - self.wall
        self.logger.info(f"Wall time: {wall}")

    def end_of_epoch_log(self):
        """
        log validation details at the end of each epoch
        """

        lr = self.optim.param_groups[0]["lr"]
        wall = perf_counter() - self.wall
        self.mae_dict = dict(
            LR=lr,
            epoch=self.iepoch,
            wall=wall,
        )

        header = "# Epoch\n# wall\n# LR"
        log_header = {
            TRAIN: "# Epoch         wall       LR",
            VALIDATION: "# Epoch         wall       LR",
        }
        mat_str = f"{self.iepoch+1:7d} {wall:12.3f} {lr:8.3g}"
        log_str = {TRAIN: f"{mat_str}", VALIDATION: f"{mat_str}"}

        for category in [TRAIN, VALIDATION]:

            stats = self.statistics[category]

            for name, stat in stats.items():

                if name == LOSS_KEY:

                    short_name = ABBREV.get(name, name)

                    arr = torch.as_tensor(stat[VALUE_KEY])
                    mean = arr.mean().item()
                    std = arr.std().item()

                    mat_str += f" {mean:8.3f} {std:8.3f}"
                    header += f"\n# {category}_{short_name}_mean\n# {category}_{short_name}_std"

                    log_header[category] += f" {short_name:>16s}"
                    log_str[category] += f" {mean:16.3g}"
                    self.mae_dict[f"{category}_{short_name}"] = mean

                for key in stat[CONTRIB]:
                    arr = torch.as_tensor(stat[CONTRIB][key])
                    mean = arr.mean().item()
                    mat_str += f" {mean:8.3f}"
                    item_name = f"{ABBREV.get(key, key)}_{ABBREV.get(name, name)}"
                    header += f"\n# {category}_{item_name}"

                    if name != LOSS_KEY:
                        log_str[category] += f" {mean:12.3f}"
                        log_header[category] += f" {item_name:>12s}"
                    self.mae_dict[f"{category}_{item_name}"] = mean

        if not self.epoch_header_print:
            self.epoch_logger.info(header)
            self.epoch_header_print = True
        self.epoch_logger.info(mat_str)

        self.logger.info("\n\n  Train      " + log_header[TRAIN])
        self.logger.info("! Train      " + log_str[TRAIN])
        self.logger.info("  Validation " + log_header[VALIDATION])
        self.logger.info("! Validation " + log_str[VALIDATION])

    def __del__(self):

        if not self.append:

            logger = self.logger
            for hdl in logger.handlers:
                hdl.flush()
                hdl.close()
            logger.handlers = []

            for i in range(len(logger.handlers)):
                logger.handlers.pop()

    def set_dataset(self, dataset):

        if self.train_idcs is None or self.val_idcs is None:

            total_n = len(dataset)

            if (self.n_train + self.n_val) > total_n:
                raise ValueError(
                    "too little data for training and validation. please reduce n_train and n_val"
                )

            if self.train_val_split == "random":
                idcs = torch.randperm(total_n)
            elif self.train_val_split == "sequential":
                idcs = torch.arange(total_n)
            else:
                raise NotImplementedError(
                    f"splitting mode {self.train_val_split} not implemented"
                )

            self.train_idcs = idcs[: self.n_train]
            self.val_idcs = idcs[self.n_train : self.n_train + self.n_val]

        # torch_geometric datasets inherantly support subsets using `index_select`
        self.dataset_train = dataset.index_select(self.train_idcs)
        self.dataset_val = dataset.index_select(self.val_idcs)

        self.dl_train, self.loader_params = instantiate(
            cls_name=DataLoader,
            prefix="loader",
            positional_args=dict(
                dataset=self.dataset_train,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                exclude_keys=self.exclude_keys,
            ),
            optional_args=self.loader_params,
            all_args=self.kwargs,
        )
        self.dl_val, _ = instantiate(
            cls_name=DataLoader,
            prefix="loader",
            positional_args=dict(
                dataset=self.dataset_val,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                exclude_keys=self.exclude_keys,
            ),
            optional_args=self.loader_params,
            all_args=self.kwargs,
        )

        self.n_train_batches = len(self.dl_train.dataset)
        self.n_val_batches = len(self.dl_val.dataset)
