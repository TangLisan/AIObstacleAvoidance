"""
This file defines the base class of all CNN models, and other models will inherit this base
"""


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from sklearn.externals import joblib
import os

from config import config
from dataset import dataset
from exception import C8Exception as c8e

import util
from asyncio.tasks import sleep
from lxml.ElementInclude import include


class model_base():
    def __init__(self, dataset = None, model = None,
                 epoch = 100, train_patience = 20, 
                 batch_size = 200, freeze_layer_number = 0,
                 image_size = (224, 224),
                 include_top = False,
                 classes = 3):
        """
        Constructor
        """
        self.dataset                    = dataset
        self.model                      = model #used to store keras type model
        self.epoch                      = epoch
        self.train_patience             = train_patience
        self.batch_size                 = batch_size
        self.freeze_layers_number       = freeze_layer_number
        self.image_size                 = image_size  
        self.optimizer                  = None 
        self.callbacks                  = None
        self.include_top                = include_top
        self.classes                    = classes
        
        
        if not os.path.exists(config.configured_trained_dir):
            os.mkdir(config.configured_trained_dir)

    def define(self, optimizer = Adam(lr=1e-5)):
        raise NotImplementedError('child class must override define()')
    
    def fineturn(self, model):
        print('No fine turn is implemented')
        
    def generate(self):
        raise NotImplementedError('child class must override generate()')
    
    def train(self):
        raise NotImplementedError('child class must override train()')
    
    def predict(self, inputs = None):
        if None == inputs:
            return [0.0, 0.0]
        
        return self.model.predict(inputs)
        
    def save(self):
        print('save operation')
    
    def load_trained_model(self):
        """Load the weight file of trained model from specified path
        """
        if (None == config.configured_model_weights) or (not os.path.exists(config.configured_model_weights)):
            raise c8e('trained weight file not exist')
        
        
        self.model.load_weights(config.configured_model_weights)
        
        print('Model load weight file successfully')
        
        return self.model
    
    def test(self):
        print('test operation') 
        
    def model_debug(self, model):
        with os.fdopen(os.open(config.configured_model_json_file, os.O_CREAT | os.O_RDWR ), 'w') as json_fd:
            json_fd.write(model.to_json())
            
            model_config = model.get_config()
      
        
    def set_trainable(self, model, flag = True):
        for layer in model.layers:
            layer.trainable = flag

    def get_input_tensor(self):
        return Input(shape = self.image_size + (3,)) #for tf, it's (224, 224, 3)
    
    def get_input_shape(self):
        return (self.image_size + (3, ))

    @staticmethod
    def make_net_layers_non_trainable(model):
        for layer in model.layers:
            layer.trainable = False

    def freeze_top_layers(self):
        if self.freeze_layers_number:
            print("Freezing {} layers".format(self.freeze_layers_number))
            for layer in self.model.layers[:self.freeze_layers_number]:
                layer.trainable = False
            for layer in self.model.layers[self.freeze_layers_number:]:
                layer.trainable = True
        else:   #added for freeze = 0
            for layer in self.model.layers:
                layer.trainable = True
       
    def force_set_all_layer_trainable(self):            
        """
        """
        for layer in self.model.layers:
            layer.trainable = True

    @staticmethod
    def get_callbacks(weights_path, patience=30, monitor='val_loss'):
        """
        generate necessary callback functions: earlyStopping, checkpoint
        """
        early_stopping = EarlyStopping(verbose=1, patience=patience, monitor=monitor)
        
        #Update: saveAllModelInformation, save_weights_only = False
        #model_checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, monitor=monitor)
        
#         return [early_stopping, model_checkpoint]
        return []

    @staticmethod
    def apply_mean(image_data_generator):
        """Subtracts the dataset mean"""
        image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))


    @staticmethod
    def save_classes():
        #joblib.dump(config.classes, config.get_classes_path())
        print('Need to support')
        
    @staticmethod
    def load_classes():
        #config.classes = joblib.load(config.get_classes_path())
        print('Need to support')


