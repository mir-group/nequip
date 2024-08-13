# Weights and Biases

The progress of loss and other metrics throughout the training process are difficult to follow without plots, and keeping track of and comparing your various trainings is difficult with ad-hoc tooling. We provide an interface to the excellent [Weights and Biases](TODO) service for this purpose.

To use `wandb`, add the following options to your config:
```yaml
wandb: true
wandb_project: my-project-name
```

You can work through a live example of using `wandb` in the [Allegro tutorial on Google Colab](https://colab.research.google.com/drive/1yq2UwnET4loJYg_Fptt9kpklVaZvoHnq).

Alternatively, we also have limited support for Tensorboard:
```yaml
tensorboard: true
```