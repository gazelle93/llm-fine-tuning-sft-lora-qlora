import os
import os.path as op


class BaseConfig(object):
    BASEDIR = op.abspath(op.dirname(__file__))
    PROJECT_ROOT = BASEDIR

    MODEL_NAME = "Qwen/Qwen1.5-1.8B"

    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "v_proj"]

    EPOCHS = 1
    BATCH_SIZE = 2
    LEARNING_RATE = 2e-5
    MAX_OUTPUT_TOKEN = 50
    TOP_P = 0.9
    TEMPERATURE = 0.7

    MAX_LENGTH = 512

    OUTPUT_DIR = "./qwen-outputs"

Config = BaseConfig