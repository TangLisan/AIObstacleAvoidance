import numpy as np
import argparse
import traceback
import os
import importlib

np.random.seed(1337)  # for reproducibility

import util

from config import config
from dataset import  dataset
from exception import C8Exception
from models import model_base
from config import env


def get_model_instance(*args, **kwargs):
    print("models.{}".format(config.model))
    
    module = importlib.import_module("models.{}".format(config.configured_model))
    
    return module.inst_class(*args, **kwargs)

def get_model_file_path():
    """
    file = 'model-{}.h5'.format(config.configured_model)
    filepath = os.path.join(config.configured_model_path,
                        file)
    """
    return config.configured_model_weights