
from keras.applications.vgg16 import VGG16 as KerasVGG16
# from .vgg16_base import VGG16_base as VGG16_base

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import os
import h5py

import numpy as np
from sklearn.externals import joblib
from config import config

#Import module of this project
from .model_base import model_base
from dataset import dataset as dt
from models import model_manager
from models import model_weight as mw

"""Define the VGG16 model
"""
class VGG16(model_base):
    noveltyDetectionLayerName = 'fc2'
    noveltyDetectionLayerSize = 4096

    def __init__(self, *args, **kwargs):   
        """Contructor
        """     
        model_base.__init__(self, *args, **kwargs)    #invoke some function of base class

    def define(self, optimizer = Adam(lr= config.configured_learning_rate)):
        """Define the model layers
        The defined model should be saved into base class
        """ 
        self.optimizer = optimizer
                
        #define the base model based on keras vgg16 model
        keras_model = KerasVGG16(weights = config.configured_pretrained_weight,
                                 include_top = False, # if config.use_fineturn_model else True, 
                                 input_tensor = self.get_input_tensor(),
                                 input_shape = self.get_input_shape(),
                                 classes = len(config.configured_classes)
                                 )
        
        self.set_trainable(keras_model, False)
        
        self.model_debug(keras_model)
        

        x = keras_model.output
        x = Flatten()(x)
        x = Dense(4096, activation='elu', name='fc1')(x)
        x = Dropout(0.6)(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.6)(x)
        predictions = Dense(len(config.configured_classes), activation='softmax', name='predictions')(x)

        self.model = Model(input=keras_model.input, output=predictions)
        
        """
        #set all layer as trainable
        self.set_trainable(keras_model, True)
        
        #use standard model or fine turn model
        if config.use_fineturn_model:
            self.model = self.fineturn(self, keras_model)
        else:
            self.model = Model(input = keras_model.input, 
                               output = keras_model.output)  #this means NO CHANGE   
        """    
    def set_weight(self):
        if not os.path.exists(config.configured_pretrained_weight):
            raise ValueError('Cannot find weight file' + config.configured_pretrained_weight)
        
        fd = h5py.File(config.configured_pretrained_weight)
        
        """
        for k in range(fd.attrs['nb_layers']):
            if k >= len(self.model.layers):
                raise ValueError('Layer count in weight isnot equal with defined model' + len(self.model.layers) + k)
            
            g = fd['layer_{}'.format(k)]
            weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
            self.model.layers[k].set_weights(weights)
        """
        
        #for i in range(len(fd.attrs.items()) - 1):
        layer_index = 0
        for layer, g in fd.items():  # 璇诲彇鍚勫眰鐨勫悕绉颁互鍙婂寘鍚眰淇℃伅鐨凣roup绫�
            self.model.layers[layer_index].set_weights(layer.get_weights())
            layer_index += 1
        
        
        fd.close()
        print('Model loaded.') 
                
    def generate(self):   
        """Compile and configure defined model
        """        
             
        #Set tainable attribute of [:freeze] to False
        self.freeze_top_layers()
        
        #set weight from pretrained weight file
        #self.set_weight()
        
        #Set all layers to trainable as True
        if config.configured_trainable:        
            self.set_trainable(self.model, True)      #set all layer of self-defined as trainable

        self.model.compile(loss = config.configured_loss,
                            optimizer = self.optimizer,        #learning rate, default = 0.001
                            metrics = config.configured_metrics)
        
        #print the overview information of this model
        self.model.summary()
        
        #Change: save checkpoint
        self.callbacks = self.get_callbacks(config.configured_model_weights, 
                                            patience = self.train_patience)
       
    
    
    def train(self):
        """Train operation
        """
        
        self.dataset.debug()            

        #train model using batched data set
        self.model.fit_generator(
            self.dataset.get_dataflow(config.configured_train_dir),
            steps_per_epoch = self.dataset.get_phase_count(config.configured_train_dir) / float(config.configured_batch_size),
            epochs = config.configured_epoch,
            validation_data = self.dataset.get_dataflow(config.configured_valid_dir),
            validation_steps = self.dataset.get_phase_count(config.configured_valid_dir) / float(config.configured_batch_size),
            callbacks = self.callbacks,
            class_weight = self.dataset.get_class_weight()
            )

        self.model.save(model_manager.get_model_file_path())
        
    def fineturn(self, keras_model):
        noveltyDetectionLayerName = 'fc2'
        noveltyDetectionLayerSize = 4096
        x = keras_model.output
        
        x = Flatten()(x)
        x = Dense(4096, activation='elu', name='fc1')(x)
        x = Dropout(0.6)(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.6)(x)
        
        predictions = Dense(len(config.configured_classes), activation='softmax', name='predictions')(x)

        return Model(input = keras_model.input, output=predictions)


def inst_class(*args, **kwargs):
    """Instrant VGG16 class
    """

        
    print('args: ', args)
    print('keyargs: ', kwargs)
    return VGG16(*args, **kwargs)
