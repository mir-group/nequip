from nequip.data import AtomicDataDict

RMSE_LOSS_KEY = "rmse"
MAE_KEY = "mae"
LOSS_KEY = "noramlized_loss"

VALUE_KEY = "value"
CONTRIB = "contrib"

VALIDATION = "Validation"
TRAIN = "Training"

ABBREV = {
    AtomicDataDict.TOTAL_ENERGY_KEY: "e",
    AtomicDataDict.FORCE_KEY: "f",
    LOSS_KEY: "loss",
    VALIDATION: "val",
}
