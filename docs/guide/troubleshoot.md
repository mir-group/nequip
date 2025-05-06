# Troubleshooting

This page covers how one can troubleshoot a variety of issues when using the NequIP framework, and will be continuously improved based on user feedback.
You are also encouraged to look through past [GitHub issues](https://github.com/mir-group/nequip/issues), [GitHub discussions](https://github.com/mir-group/nequip/discussions), and our [FAQ page](faq.md).

## Poor Training Behavior

If you're model doesn't seem like it's learning, the reasons could range from problematic model hyperparameters, to problematic training hyperparameters, to data-side problems.

As a start, ensure that the data format conforms to [NequIP conventions](conventions.md). For example, a common failure mode is when a dataset uses the opposite sign convention for stress.

In terms, of training hyperparameters, NequIP models tend to learn with learning rates of around 0.01, while Allegro models are better with learning rates of around 0.001.
One can always try to lower the learning rate to test if the problem could lie with a large learning rate causing large jumps in the loss landscape.


## Commons Errors

### `nequip-train config.yaml` fails

  **Problem**: Trying to run `nequip-train` as follows fails.
```bash
nequip-train config.yaml
```
  **Solution**: Read the [workflow docs](workflow.md) and follow hydra's command line options, e.g.
```bash
nequip-train -cn config.yaml
```
