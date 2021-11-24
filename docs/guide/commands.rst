Command-line Reference
======================

``nequip-train``
----------------

 .. code ::

    usage: nequip-train [-h] [--equivariance-test] [--model-debug-mode] [--grad-anomaly-mode] [--log LOG] config

Train (or restart training of) a NequIP model.

positional arguments:
  config               YAML file configuring the model, dataset, and other options

optional arguments:
  -h, --help           show this help message and exit
  --equivariance-test  test the model's equivariance before training
  --model-debug-mode   enable model debug mode, which can sometimes give much more useful error messages at the
                       cost of some speed. Do not use for production training!
  --grad-anomaly-mode  enable PyTorch autograd anomaly mode to debug NaN gradients. Do not use for production
                       training!
  --log LOG            log file to store all the screen logging

``nequip-evaluate``
-------------------

 .. code ::

    usage: nequip-evaluate [-h] [--train-dir TRAIN_DIR] [--model MODEL] [--dataset-config DATASET_CONFIG]
                        [--metrics-config METRICS_CONFIG] [--test-indexes TEST_INDEXES] [--batch-size BATCH_SIZE]
                        [--device DEVICE] [--output OUTPUT] [--log LOG]

Compute the error of a model on a test set using various metrics. The model, metrics, dataset, etc. can specified
in individual YAML config files, or a training session can be indicated with ``--train-dir``. In order of priority,
the global settings (dtype, TensorFloat32, etc.) are taken from: (1) the model config (for a training session), (2)
the dataset config (for a deployed model), or (3) the defaults. Prints only the final result in ``name = num`` format
to stdout; all other information is ``logging.debug``ed to stderr. WARNING: Please note that results of CUDA models
are rarely exactly reproducible, and that even CPU models can be nondeterministic.

optional arguments:
  -h, --help            show this help message and exit
  --train-dir TRAIN_DIR
                        Path to a working directory from a training session.
  --model MODEL         A deployed or pickled NequIP model to load. If omitted, defaults to `best_model.pth` in
                        `train_dir`.
  --dataset-config DATASET_CONFIG
                        A YAML config file specifying the dataset to load test data from. If omitted, `config.yaml`
                        in `train_dir` will be used
  --metrics-config METRICS_CONFIG
                        A YAML config file specifying the metrics to compute. If omitted, `config.yaml` in
                        `train_dir` will be used. If the config does not specify `metrics_components`, the default
                        is to logging.debug MAEs and RMSEs for all fields given in the loss function. If the
                        literal string `None`, no metrics will be computed.
  --test-indexes TEST_INDEXES
                        Path to a file containing the indexes in the dataset that make up the test set. If omitted,
                        all data frames *not* used as training or validation data in the training session
                        `train_dir` will be used.
  --batch-size BATCH_SIZE
                        Batch size to use. Larger is usually faster on GPU.
  --device DEVICE       Device to run the model on. If not provided, defaults to CUDA if available and CPU
                        otherwise.
  --output OUTPUT       XYZ file to write out the test set and model predicted forces, energies, etc. to.
  --log LOG             log file to store all the metrics and screen logging.debug

``nequip-deploy``
-----------------

 .. code ::

    usage: nequip-deploy [-h] {info,build} ...

Deploy and view information about previously deployed NequIP models.

optional arguments:
  -h, --help    show this help message and exit

commands:
  {info,build}
    info        Get information from a deployed model file
    build       Build a deployment model

``nequip-deploy info``
~~~~~~~~~~~~~~~~~~~~~~

 .. code ::

    usage: nequip-deploy info [-h] model_path

positional arguments:
  model_path  Path to a deployed model file.

optional arguments:
  -h, --help  show this help message and exit


``nequip-deploy build``
~~~~~~~~~~~~~~~~~~~~~~~

 .. code ::

    usage: nequip-deploy build [-h] train_dir out_file

positional arguments:
  train_dir   Path to a working directory from a training session.
  out_file    Output file for deployed model.

optional arguments:
  -h, --help  show this help message and exit


``nequip-benchmark``
--------------------

 .. code ::

    usage: nequip-benchmark [-h] [--profile PROFILE] [--device DEVICE] [-n N] [--n-data N_DATA] [--timestep TIMESTEP]
                            config

Benchmark the approximate MD performance of a given model configuration / dataset pair.

positional arguments:
  config               configuration file

optional arguments:
  -h, --help           show this help message and exit
  --profile PROFILE    Profile instead of timing, creating and outputing a Chrome trace JSON to the given path.
  --device DEVICE      Device to run the model on. If not provided, defaults to CUDA if available and CPU
                       otherwise.
  -n N                 Number of trials.
  --n-data N_DATA      Number of frames to use.
  --timestep TIMESTEP  MD timestep for ns/day esimation, in fs. Defauts to 1fs.
