from nequip.data import AtomicDataDict

RMSE_LOSS_KEY = "rmse"
MAE_KEY = "mae"
LOSS_KEY = "noramlized_loss"

VALUE_KEY = "value"
CONTRIB = "contrib"

VALIDATION = "validation"
TRAIN = "training"

ABBREV = {
    AtomicDataDict.TOTAL_ENERGY_KEY: "e",
    AtomicDataDict.PER_ATOM_ENERGY_KEY: "Ei",
    AtomicDataDict.FORCE_KEY: "f",
    LOSS_KEY: "loss",
    VALIDATION: "val",
    TRAIN: "train",
}
