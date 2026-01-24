# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
SEQ_LEN = 128
LR = 3e-4
WEIGHT_DECAY = 0.01

# Model hyperparameters
D_MODEL = 512
H = 8  # Number of attention heads
N = 6  # Number of decoder layers
D_FF = 2048  # Feed-forward dimension
DROPOUT = 0.1

# Training settings
VAL_SPLIT = 0.1
MODEL_FOLDER = 'weights'
EXPERIMENT_NAME = 'runs/language_model'
USE_MIXED_PRECISION = True
PRELOAD = None 