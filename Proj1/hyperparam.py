LR = [5e-2, 5e-3, 5e-4]
WEIGHT_DECAY = [1e-3, 1e-4, 1e-5]  # L2 penalty of the Adam optimizer
BATCH_SIZE = 100
# Coefficient to combine auxiliary losses with true loss
AUX_PARAM = [.2, .4, .6]
SAMPLE_SIZE = 1000

HYPERPARAMS = {
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": BATCH_SIZE,
    "aux_param": AUX_PARAM,
    "sample_size": SAMPLE_SIZE,
}
