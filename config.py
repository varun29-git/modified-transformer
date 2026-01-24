# Training hyperparameters - A100 5GB OPTIMIZED
EPOCHS = 5
BATCH_SIZE = 32  # Safer for 5GB
SEQ_LEN = 256
LR = 3e-4
WEIGHT_DECAY = 0.01

# Model hyperparameters
D_MODEL = 384
H = 6
N = 6
D_FF = 1536
DROPOUT = 0.1

# Training settings
VAL_SPLIT = 0.05
MODEL_FOLDER = 'weights'
EXPERIMENT_NAME = 'runs/tinystories'
USE_MIXED_PRECISION = True
PRELOAD = None