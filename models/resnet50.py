from keras.applications.resnet50 import ResNet50 as KerasResNet50
from keras.layers import (Flatten, Dense, Dropout)
from keras.optimizers import Adam
from keras.models import Model

from .model_base import model_base
from dataset import dataset as dt
from models import model_manager
from models import model_weight as mw


import numpy as np
from sklearn.externals import joblib
from config import config

class ResNet50(model_base):
    noveltyDetectionLayerName = 'fc1'
    noveltyDetectionLayerSize = 2048

    def __init__(self, *args, **kwargs):
        model_base.__init__(self, *args, **kwargs)
        
        """
        if not self.freeze_layers_number:
            # we chose to train the top 2 identity blocks and 1 convolution block
            self.freeze_layers_number = 80
        """
        
    def define(self, optimizer = Adam(lr= config.configured_learning_rate)):
        
        self.optimizer = optimizer
        
        keras_model = KerasResNet50(weights = None,
                                    include_top=True, 
                                    input_tensor=self.get_input_tensor(),
                                    input_shape = self.get_input_shape(),
                                    classes = len(config.configured_classes))
        
        self.make_net_layers_non_trainable(keras_model)
        
        #use standard model or fine turn model
        if config.use_fineturn_model:
            self.model = self.fineturn(self, keras_model)
        else:
            self.model = Model(input = keras_model.input, 
                               output = keras_model.output)  #this means NO CHANGE   
            
            
                
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

        x = keras_model.output
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        # we could achieve almost the same accuracy without this layer, buy this one helps later
        # for novelty detection part and brings much more useful features.
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(config.configured_classes), activation='softmax', name='predictions')(x)

        self.model = Model(input=keras_model.input, output=predictions)


def inst_class(*args, **kwargs):
    return ResNet50(*args, **kwargs)
