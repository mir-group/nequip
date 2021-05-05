""" Nequip.train.trainer

Todo:

isolate the loss function from the training procedure
enable wandb resume
make an interface with ray

"""
import sys
import inspect
import logging
import yaml
from copy import deepcopy
from os.path import isfile
from time import perf_counter
from typing import Optional, Union

if sys.version_info[1] >= 7:
    import contextlib
else:
    # has backport of nullcontext
    import contextlib2 as contextlib

import numpy as np
import torch
from torch_ema import ExponentialMovingAverage

from nequip.data import DataLoader, AtomicData, AtomicDataDict
from nequip.utils import (
    Output,
    instantiate_from_cls_name,
    instantiate,
    save_file,
    load_file,
    atomic_write,
)

from .loss import Loss, LossStat
from .metrics import Metrics
from ._key import ABBREV, LOSS_KEY, TRAIN, VALIDATION


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

    For a fresh run, the simulation results will be stored in the 'root/run_name' folder. With
    - "log" : plain text information about the whole training process
    - "metrics_epoch.csv" : txt matrice format (readable by np.loadtxt) with loss/mae from training and validation of each epoch
    - "metrics_batch.csv" : txt matrice format (readable by np.loadtxt) with training loss/mae of each batch
    - "best_model.pth": the best model. save the model when validation mae goes lower
    - "last_model.pth": the last model. save the model every log_epoch_freq epoch
    - "trainer_save.pth": all the training information. The file used for loading and restart

    For restart run, the default set up is to not append to the original folders and files.
    The Output class will automatically build a folder call root/run_name
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

        run_name (str): run name.
        root (str): the name of root dir to make work folders
        timestr (optional, str): unique string to differentiate this trainer from others.

        restart (bool) : If true, the init_model function will not be callsed. Default: False
        append (bool): If true, the preexisted workfolder and files will be overwritten. And log files will be appended

        loss_coeffs (dict): dictionary to store coefficient and loss functions

        max_epochs (int): maximum number of epochs

        lr_sched (optional): scheduler
        learning_rate (float): initial learning rate
        lr_scheduler_name (str): scheduler name
        lr_scheduler_kwargs (dict): parameters to initialize the scheduler

        optim (): optimizer
        optimizer_name (str): name for optimizer
        optim_kwargs (dict): parameters to initialize the optimizer

        batch_size (int): size of each batch
        shuffle (bool): parameters for dataloader
        n_train (int): # of frames for training
        n_val (int): # of frames for validation
        exclude_keys (list):  fields from dataset to ignore.
        dataloader_num_workers (int): `num_workers` for the `DataLoader`s
        train_idcs (optional, list):  list of frames to use for training
        val_idcs (list):  list of frames to use for validation
        train_val_split (str):  "random" or "sequential"

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

    lr_scheduler_module = torch.optim.lr_scheduler
    optim_module = torch.optim

    def __init__(
        self,
        model,
        run_name: Optional[str] = None,
        root: Optional[str] = None,
        timestr: Optional[str] = None,
        seed: Optional[int] = None,
        restart: bool = False,
        append: bool = True,
        loss_coeffs: Union[dict, str] = AtomicDataDict.TOTAL_ENERGY_KEY,
        metrics_components: Optional[Union[dict, str]] = None,
        metrics_key: str = ABBREV.get(LOSS_KEY, LOSS_KEY),
        early_stop_lower_threshold: Optional[float] = None,
        max_epochs: int = 1000000,
        lr_sched=None,
        learning_rate: float = 1e-2,
        lr_scheduler_name: str = "none",
        lr_scheduler_kwargs: Optional[dict] = None,
        optim=None,
        optimizer_name: str = "Adam",
        optimizer_kwargs: Optional[dict] = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_use_num_updates=True,
        exclude_keys: list = [],
        batch_size: int = 5,
        shuffle: bool = True,
        n_train: Optional[int] = None,
        n_val: Optional[int] = None,
        dataloader_num_workers: int = 0,
        train_idcs: Optional[list] = None,
        val_idcs: Optional[list] = None,
        train_val_split: str = "random",
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

        _local_kwargs = {}
        for key in self.init_params:
            setattr(self, key, locals()[key])
            _local_kwargs[key] = locals()[key]

        if self.use_ema:
            self.ema = None

        output = Output.get_output(timestr, dict(**_local_kwargs, **kwargs))
        self.output = output

        # timestr run_name root workdir logfile
        for key, value in output.updated_dict().items():
            setattr(self, key, value)

        if self.logfile is None:
            self.logfile = output.open_logfile("log", propagate=True)
        self.epoch_log = output.open_logfile("metrics_epoch.csv", propagate=False)
        self.batch_log = {
            TRAIN: output.open_logfile("metrics_batch_train.csv", propagate=False),
            VALIDATION: output.open_logfile("metrics_batch_val.csv", propagate=False),
        }

        # add filenames if not defined
        self.best_model_path = output.generate_file("best_model.pth")
        self.last_model_path = output.generate_file("last_model.pth")
        self.trainer_save_path = output.generate_file("trainer.pth")

        if not (seed is None or self.restart):
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Torch device: {self.device}")

        # sort out all the other parameters
        # for samplers, optimizer and scheduler
        self.kwargs = deepcopy(kwargs)
        self.optimizer_kwargs = deepcopy(optimizer_kwargs)
        self.lr_scheduler_kwargs = deepcopy(lr_scheduler_kwargs)

        # initialize the optimizer and scheduler, the params will be updated in the function
        self.init()

        self.statistics = {}

        if not (restart and append):
            d = self.as_dict()
            for key in list(d.keys()):
                if not isinstance(d[key], (float, int, str, list, tuple)):
                    d[key] = type(d[key])
            self.log_dictionary(d, name="Initialization")

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

        state_dict (bool): if True, the state_dicts of the optimizer, lr scheduler, and EMA will be included
        """

        dictionary = {}

        for key in self.init_params:
            dictionary[key] = getattr(self, key, None)
        dictionary.update(getattr(self, "kwargs", {}))

        if state_dict:
            dictionary["state_dict"] = {}
            dictionary["state_dict"]["optim"] = self.optim.state_dict()
            if self.lr_sched is not None:
                dictionary["state_dict"]["lr_sched"] = self.lr_sched.state_dict()
            dictionary["state_dict"]["rng_state"] = torch.get_rng_state()
            if torch.cuda.is_available():
                dictionary["state_dict"]["cuda_rng_state"] = torch.cuda.get_rng_state(
                    device=self.device
                )
            if self.use_ema:
                dictionary["state_dict"]["ema_state"] = self.ema.state_dict()

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

            # TODO: these might not both be available, str defined, but no weights
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
        logger.debug(f"Saved trainer to {filename}")

        with atomic_write(self.last_model_path) as write_to:
            if hasattr(self.model, "save"):
                self.model.save(write_to)
            else:
                torch.save(self.model, write_to)
        logger.debug(f"Saved last model to to {self.last_model_path}")

        return filename

    @classmethod
    def from_file(
        cls, filename: str, format: Optional[str] = None, append: Optional[bool] = None
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
        return cls.from_dict(dictionary, append)

    @classmethod
    def from_dict(cls, dictionary, append: Optional[bool] = None):
        """load model from dictionary

        Args:

        dictionary (dict):
        append (bool): if True, append the old model files and append the same logfile
        """

        d = deepcopy(dictionary)

        # update the restart and append option
        d["restart"] = True
        if append is not None:
            d["append"] = append

        # update the file and folder name
        output = Output.from_config(d)
        d.update(output.updated_dict())

        model = None
        iepoch = 0
        if "model" in d:
            model = d.pop("model")
        elif "progress" in d:
            progress = d["progress"]
            stop_arg = progress.pop("stop_arg", None)
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
            else:
                raise AttributeError("model weights & bias are not saved")

            if "model_class" in d:
                model = d["model_class"].load(load_path)
            else:
                if dictionary.get("compile_model", False):
                    model = torch.jit.load(load_path)
                else:
                    model = torch.load(load_path)
            logging.debug(f"Reload the model from {load_path}")

            d.pop("progress")

        state_dict = d.pop("state_dict", None)

        trainer = cls(model=model, **d)

        if state_dict is not None and trainer.model is not None:
            logging.debug("Reload optimizer and scheduler states")
            trainer.optim.load_state_dict(state_dict["optim"])

            if trainer.lr_sched is not None:
                trainer.lr_sched.load_state_dict(state_dict["lr_sched"])

            torch.set_rng_state(state_dict["rng_state"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(state_dict["cuda_rng_state"])

            if trainer.use_ema:
                trainer.ema.load_state_dict(state_dict["ema_state"])

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

        self.model.to(self.device)

        if self.optim is None:
            self.optim, self.optimizer_kwargs = instantiate_from_cls_name(
                module=torch.optim,
                class_name=self.optimizer_name,
                prefix="optimizer",
                positional_args=dict(
                    params=self.model.parameters(), lr=self.learning_rate
                ),
                all_args=self.kwargs,
                optional_args=self.optimizer_kwargs,
            )

        if self.use_ema and self.ema is None:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                decay=self.ema_decay,
                use_num_updates=self.ema_use_num_updates,
            )

        if self.lr_sched is None:
            assert (
                self.lr_scheduler_name
                in ["CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "none"]
            ) or (
                (len(self.end_of_epoch_callbacks) + len(self.end_of_batch_callbacks))
                > 0
            ), f"{self.lr_scheduler_name} cannot be used unless callback functions are defined"
            self.lr_sched = None
            self.lr_scheduler_kwargs = {}
            if self.lr_scheduler_name != "none":
                self.lr_sched, self.lr_scheduler_kwargs = instantiate_from_cls_name(
                    module=torch.optim.lr_scheduler,
                    class_name=self.lr_scheduler_name,
                    prefix="lr_scheduler",
                    positional_args=dict(optimizer=self.optim),
                    optional_args=self.lr_scheduler_kwargs,
                    all_args=self.kwargs,
                )

        self.loss, _ = instantiate(
            builder=Loss,
            prefix="loss",
            positional_args=dict(coeffs=self.loss_coeffs),
            all_args=self.kwargs,
        )
        self.loss_stat = LossStat(keys=list(self.loss.funcs.keys()))
        self._initialized = True

    def init_metrics(self):
        if self.metrics_components is None:
            self.metrics_components = []
            for key, func in self.loss.funcs.items():
                params = {
                    "PerSpecies": type(func).__name__.startswith("PerSpecies"),
                }
                self.metrics_components.append((key, "mae", params))
                self.metrics_components.append((key, "rmse", params))

        self.metrics, _ = instantiate(
            builder=Metrics,
            prefix="metrics",
            positional_args=dict(components=self.metrics_components),
            all_args=self.kwargs,
        )

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
        self.init_metrics()

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

    def batch_step(self, data, validation=False):
        # no need to have gradients from old steps taking up memory
        self.optim.zero_grad(set_to_none=True)

        if validation:
            self.model.eval()
        else:
            self.model.train()

        # Do any target rescaling
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        if hasattr(self.model, "unscale"):
            # This means that self.model is RescaleOutputs
            # this will normalize the targets
            # in validation (eval mode), it does nothing
            # in train mode, if normalizes the targets
            data_unscaled = self.model.unscale(data)
        else:
            data_unscaled = data

        # Run model
        # We make a shallow copy of the input dict in case the model modifies it
        input_data = data_unscaled.copy()
        out = self.model(input_data)
        del input_data

        # If we're in evaluation mode (i.e. validation), then
        # data_unscaled's target prop is unnormalized, and out's has been rescaled to be in the same units
        # If we're in training, data_unscaled's target prop has been normalized, and out's hasn't been touched, so they're both in normalized units
        # Note that either way all normalization was handled internally by RescaleOutput

        if not validation:
            loss, loss_contrib = self.loss(pred=out, ref=data_unscaled)
            # see https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()

            if self.use_ema:
                self.ema.update()

            if self.lr_scheduler_name == "CosineAnnealingWarmRestarts":
                self.lr_sched.step(self.iepoch + self.ibatch / self.n_batches)

        with torch.no_grad():
            if hasattr(self.model, "unscale"):
                if validation:
                    # loss function always needs to be in normalized unit
                    scaled_out = self.model.unscale(out, force_process=True)
                    _data_unscaled = self.model.unscale(data, force_process=True)
                    loss, loss_contrib = self.loss(pred=scaled_out, ref=_data_unscaled)
                else:
                    # If we are in training mode, we need to bring the prediction
                    # into real units
                    out = self.model.scale(out, force_process=True)
            elif validation:
                loss, loss_contrib = self.loss(pred=out, ref=data_unscaled)

            # save metrics stats
            self.batch_losses = self.loss_stat(loss, loss_contrib)
            # in validation mode, data is in real units and the network scales
            # out to be in real units interally.
            # in training mode, data is still in real units, and we rescaled
            # out to be in real units above.
            self.batch_metrics = self.metrics(pred=out, ref=data)

    @property
    def early_stop_cond(self):
        """ kill the training early """

        if self.early_stop_lower_threshold is not None:
            if self.best_val_metrics < self.early_stop_lower_threshold:
                return True
        return False

    def reset_metrics(self):
        self.loss_stat.reset()
        self.loss_stat.to(self.device)
        self.metrics.reset()
        self.metrics.to(self.device)

    def epoch_step(self):
        datasets = [self.dl_train, self.dl_val]
        categories = [TRAIN, VALIDATION]
        self.metrics_dict = {}
        self.loss_dict = {}

        for category, dataset in zip(categories, datasets):
            if category == VALIDATION and self.use_ema:
                cm = self.ema.average_parameters()
            else:
                cm = contextlib.nullcontext()

            with cm:
                self.reset_metrics()
                self.n_batches = len(dataset)
                for self.ibatch, batch in enumerate(dataset):
                    self.batch_step(
                        data=batch,
                        validation=(category == VALIDATION),
                    )
                    self.end_of_batch_log(batch_type=category)
                    for callback in self.end_of_batch_callbacks:
                        callback(self)
                self.metrics_dict[category] = self.metrics.current_result()
                self.loss_dict[category] = self.loss_stat.current_result()

                if category == TRAIN:
                    for callback in self.end_of_train_callbacks:
                        callback(self)

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

    def end_of_batch_log(self, batch_type: str):
        """
        store all the loss/mae of each batch
        """

        mat_str = f"{self.iepoch+1:5d}, {self.ibatch+1:5d}"
        log_str = f"{self.iepoch+1:5d} {self.ibatch+1:5d}"

        header = "epoch, batch"
        log_header = "# Epoch batch"

        # print and store loss value
        for name, value in self.batch_losses.items():
            mat_str += f", {value:16.5g}"
            header += f", {name}"
            log_str += f" {value:12.3g}"
            log_header += f" {name:>12s}"

        # append details from metrics
        metrics, skip_keys = self.metrics.flatten_metrics(
            metrics=self.batch_metrics,
            allowed_species=self.model.config.get("allowed_species", None)
            if hasattr(self.model, "config")
            else None,
        )
        for key, value in metrics.items():

            mat_str += f", {value:16.5g}"
            header += f", {key}"
            if key not in skip_keys:
                log_str += f" {value:12.3g}"
                log_header += f" {key:>12s}"

        batch_logger = logging.getLogger(self.batch_log[batch_type])
        if not self.batch_header_print[batch_type]:
            self.batch_header_print[batch_type] = True
            batch_logger.info(header)

        if self.ibatch == 0:
            self.logger.info("")
            self.logger.info(f"{batch_type}")
            self.logger.info(log_header)

        batch_logger.info(mat_str)
        if (self.ibatch + 1) % self.log_batch_freq == 0 or (
            self.ibatch + 1
        ) == self.n_batches:
            self.logger.info(log_str)

    def end_of_epoch_save(self):
        """
        save model and trainer details
        """

        val_metrics = self.mae_dict[f"{VALIDATION}_{self.metrics_key}"]
        if val_metrics < self.best_val_metrics:
            self.best_val_metrics = val_metrics
            self.best_epoch = self.iepoch

            if self.use_ema:
                # If using EMA, store the EMA validation model
                # that gave us the good val metrics that made the model "best"
                # in the first place
                cm = self.ema.average_parameters()
            else:
                # otherwise, do nothing
                cm = contextlib.nullcontext()

            with cm:
                with atomic_write(self.best_model_path) as save_path:
                    if hasattr(self.model, "save"):
                        self.model.save(save_path)
                    else:
                        torch.save(self.model, save_path)

            self.logger.info(
                f"! Best model {self.best_epoch+1:8d} {self.best_val_metrics:8.3f}"
            )

        if (self.iepoch + 1) % self.log_epoch_freq == 0:
            self.save(self.trainer_save_path)

    def init_log(self):

        if self.restart:
            self.logger.info("! Restarting training ...")
        else:
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

        header = "epoch, wall, LR"

        categories = [TRAIN, VALIDATION]
        log_header = {}
        log_str = {}

        strings = ["Epoch", "wal", "LR"]
        mat_str = f"{self.iepoch+1:10d}, {wall:8.3f}, {lr:8.3g}"
        for cat in categories:
            log_header[cat] = "# "
            log_header[cat] += " ".join([f"{s:>8s}" for s in strings])
            log_str[cat] = f"{self.iepoch+1:10d} {wall:8.3f} {lr:8.3g}"

        for category in categories:

            met, skip_keys = self.metrics.flatten_metrics(
                metrics=self.metrics_dict[category],
                allowed_species=self.model.config.get("allowed_species", None)
                if hasattr(self.model, "config")
                else None,
            )

            # append details from loss
            for key, value in self.loss_dict[category].items():
                mat_str += f", {value:16.5g}"
                header += f", {category}_{key}"
                log_str[category] += f" {value:12.3g}"
                log_header[category] += f" {key:>12s}"
                self.mae_dict[f"{category}_{key}"] = value

            # append details from metrics
            for key, value in met.items():
                mat_str += f", {value:12.3g}"
                header += f", {category}_{key}"
                if key not in skip_keys:
                    log_str[category] += f" {value:12.3g}"
                    log_header[category] += f" {key:>12s}"
                self.mae_dict[f"{category}_{key}"] = value

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

        # based on recommendations from
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation
        if self.dataloader_num_workers != 0:
            # some issues with timeouts need to be investigated
            raise NotImplementedError
        dl_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            exclude_keys=self.exclude_keys,
            # num_workers=self.dataloader_num_workers,
            # persistent_workers=(self.max_epochs > 1),
            pin_memory=(self.device != torch.device("cpu")),
            # timeout=10,  # just so you don't get stuck
        )
        self.dl_train = DataLoader(dataset=self.dataset_train, **dl_kwargs)
        self.dl_val = DataLoader(dataset=self.dataset_val, **dl_kwargs)
