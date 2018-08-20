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



"""
    why override ?
"""
def override_keras_directory_iterator_next():
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



def get_activation_function(m, layer):
    x = [m.layers[0].input, K.learning_phase()]
    y = [m.get_layer(layer).output]
    return K.function(x, y)


def get_activations(activation_function, X_batch):
    activations = activation_function([X_batch, 0])
    return activations[0][0]


def save_activations(model, inputs, files, layer, batch_number):
    all_activations = []
    ids = []
    af = get_activation_function(model, layer)
    for i in range(len(inputs)):
        acts = get_activations(af, [inputs[i]])
        all_activations.append(acts)
        ids.append(files[i].split('/')[-2])

    submission = pd.DataFrame(all_activations)
    submission.insert(0, 'class', ids)
    submission.reset_index()
    if batch_number > 0:
        submission.to_csv(config.activations_path, index=False, mode='a', header=False)
    else:
        submission.to_csv(config.activations_path, index=False)




def get_keras_backend_name():
    try:
        return K.backend()
    except AttributeError:
        return K._BACKEND


def tf_allow_growth():
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True  #Be carefule
    sess = tf.Session(config=tf_config)
    set_session(sess)


def set_img_format():
    try:
        if K.backend() == 'theano':
            K.set_image_data_format('channels_first')
        else:
            K.set_image_data_format('channels_last')    #tf as backend
    except AttributeError:
        if K._BACKEND == 'theano':
            K.set_image_dim_ordering('th')
        else:
            K.set_image_dim_ordering('tf')
