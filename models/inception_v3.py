from keras.applications.inception_v3 import InceptionV3 as KerasInceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
import numpy as np
from keras.optimizers import Adam
from keras.models import Model

from .model_base import model_base
from dataset import dataset as dt
from models import model_manager
from models import model_weight as mw


import numpy as np
from sklearn.externals import joblib
from config import config

class InceptionV3(model_base):
    noveltyDetectionLayerName = 'fc1'
    noveltyDetectionLayerSize = 1024

    def __init__(self, *args, **kwargs):
        model_base.__init__(self, *args, **kwargs)

        """
        if not self.freeze_layers_number:
            # we chose to train the top 2 identity blocks and 1 convolution block
            self.freeze_layers_number = 80
        """
        self.img_size = (299, 299)
        
        
        
    def define(self, optimizer = Adam(lr= config.configured_learning_rate)):
        
        self.optimizer = optimizer
        
        
        keras_model = KerasInceptionV3(weights=None, 
                                       include_top=False, 
                                       input_tensor=self.get_input_tensor())
        self.make_net_layers_non_trainable(keras_model)
        
        #use standard model or fine turn model
        self.fineturn(keras_model)
       

    def fineturn(self, keras_model):
        x = keras_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.noveltyDetectionLayerSize, 
                  activation='elu', 
                  name=self.noveltyDetectionLayerName)(x)
                  
        predictions = Dense(len(config.configured_classes), 
                            activation='softmax')(x)

        self.model = Model(input = keras_model.input, output = predictions)
            
            
                
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
                            optimizer=SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True),        #learning rate, default = 0.001
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
        

    def preprocess_input(self, x):
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def load_img(self, img_path):
        img = image.load_img(img_path, target_size=self.img_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return self.preprocess_input(x)[0]

    @staticmethod
    def apply_mean(image_data_generator):
        pass


def inst_class(*args, **kwargs):
    return InceptionV3(*args, **kwargs)
