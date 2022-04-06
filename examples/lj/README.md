Run commands with
```
PYTHONPATH=`pwd`:$PYTHONPATH nequip-* ...
```
so that the model from `lj.py` can be imported.

For example, to create a deployed LJ model `lj.pth`:
```bash
PYTHONPATH=`pwd`:$PYTHONPATH nequip-deploy build --model lj.yaml lj.pth
```