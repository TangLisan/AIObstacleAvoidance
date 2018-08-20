import matplotlib; matplotlib.use('Agg')  # fixes issue if no GUI provided
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd
import importlib
import keras
from keras import backend as K
import config
import math
import itertools
from inspect import currentframe, getframeinfo

from config import config



def tf_allow_growth():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True  #Be carefule
    sess = tf.Session(config=tf_config)
    set_session(sess)


def config_img_format():  
    K.set_image_data_format('channels_last')    #tf as backend
    K.set_image_dim_ordering('tf')
    
    
def config_directory_iterator():
    """Overrides .next method of DirectoryIterator in Keras
      to reorder color channels for images from RGB to BGR"""
    from keras.preprocessing.image import DirectoryIterator

    original_next = DirectoryIterator.next

    # do not allow to override one more time
    if 'custom_next' in str(original_next):
        return

    def custom_next(self):
        batch_x, batch_y = original_next(self)
        batch_x = batch_x[:, ::-1, :, :]
        return batch_x, batch_y

    DirectoryIterator.next = custom_next
    
def config_env():
    config_img_format()
    
    config_directory_iterator()
    
    tf_allow_growth()
    
def check_env():
    
    #check directory
    if not os.path.exists(config.configured_debug_dir):
        os.mkdir(config.configured_debug_dir)
        
    if not os.path.exists(config.configured_trained_dir):
        os.mkdir(config.configured_trained_dir)
    
    
    