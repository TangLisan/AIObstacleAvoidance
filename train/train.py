"""
This file is used to provide train related function and entry point
"""
import numpy as np
import argparse
import traceback
import os

np.random.seed(1337)  # for reproducibility

import util

from config import config
from dataset import  dataset
from exception import C8Exception
from models import model_base
from config import env
from models import model_manager


def train():
    try:
        #check environment
        env.config_env()
        
        #check dataset
        dt = dataset.dataset()
        dt.check()
        dt.generate_dataflow(image_size = config.configured_image_size)
        
        dt.debug()  #for debuging data set
        
        #instance and initialize  model
        model = model_manager.get_model_instance(dataset = dt)      
        model.define()
        model.generate()
        model.train()
        
    except Exception as e:
        print(e)
        traceback.print_exc()
    finally:
        print('over')

if __name__ == '__main__':
    train()