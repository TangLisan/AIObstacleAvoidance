from keras.applications.vgg16 import VGG16 as KerasVGG16
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from sklearn.externals import joblib
from config import config
from .model_base import model_base
from dataset import dataset as dt
from models import model_manager

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras import optimizers,regularizers
from keras.callbacks import LearningRateScheduler

from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import matplotlib.pyplot as plt

#
# configured_vgg16_weights='C:/Dev/keras_weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'


epochs = 5                                        
batch_size = 32                                                  
learning_rate = float(10**-np.random.randint(0, 4)*np.random.random(1))        
learning_rate = 0.00070110810387954
lr_decay = np.random.random(1)*10**-np.random.randint(0, 2)
lr_decay = 0.1
l2_reg_weight = 0.001
decay_step = 1    



class AlexNetSJTU(model_base):
    noveltyDetectionLayerName = 'fc2'
    noveltyDetectionLayerSize = 4096

    def __init__(self, *args, **kwargs):        
        model_base.__init__(self, *args, **kwargs)    #invoke some function of base class

    def define(self, optimizer = Adam(lr=1e-5)):
        """Define the model layers
        """ 
        
        self.optimizer = optimizer

        # pretrained deep learning model
        # Incep_model = InceptionResNetV2(weights='imagenet', input_shape=(img_width, img_height, 3), include_top=False)
        # for layer in Incep_model.layers:  # do not train the pretrained model for memory saving
        #     layer.trainable = False
        
        # features1 = Incep_model.output
        # features2 = Flatten(name='Flatten')(features1)
        # features3 = Dense(256, activation='relu',kernel_regularizer=)(features2)
        # features4 = Dense(256, activation='relu')(features3)
        # features4 = Dense(256, activation='relu')(features3)
        # Incep_output = Dense(4, activation='sigmoid')(features4)
        # model = keras.models.Model(inputs=Incep_model.input, outputs=Incep_output, name='model')  # 512*512*3 -- Batch * 4
        
        model = Sequential()              
                                                  # self build deep learning model
        model.add(Conv2D(64, (3, 3), input_shape=self.get_input_shape(), 
                         kernel_regularizer=regularizers.l2(l2_reg_weight),
                         bias_regularizer=regularizers.l2(l2_reg_weight), 
                         activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(l2_reg_weight),
                         bias_regularizer=regularizers.l2(l2_reg_weight), activation='relu',padding='same'))
        model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(l2_reg_weight),
                         bias_regularizer=regularizers.l2(l2_reg_weight), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        print('model.output_shape:', model.output_shape)
        model.add(Conv2D(16, (5, 5), kernel_regularizer=regularizers.l2(l2_reg_weight),
                         bias_regularizer=regularizers.l2(l2_reg_weight), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        print('model.output_shape:', model.output_shape)
        
        model.add(Flatten())
        model.add(Dense(1024, kernel_regularizer=regularizers.l2(l2_reg_weight),
                         bias_regularizer=regularizers.l2(l2_reg_weight), activation='relu'))
        
        model.add(Dense(512, kernel_regularizer=regularizers.l2(l2_reg_weight),
                         bias_regularizer=regularizers.l2(l2_reg_weight), activation='relu'))
        
        # model.add(Dense(512, kernel_regularizer=regularizers.l2(l2_reg_weight),
        #                  bias_regularizer=regularizers.l2(l2_reg_weight), activation='relu'))
        # model.add(Dense(256, kernel_regularizer=regularizers.l2(l2_reg_weight),
        #                  bias_regularizer=regularizers.l2(l2_reg_weight), activation='relu'))
        model.add(Dense(256, kernel_regularizer=regularizers.l2(l2_reg_weight),
                         bias_regularizer=regularizers.l2(l2_reg_weight), activation='relu'))
        #model.add(Dropout(0.5))
        
        model.add(Dense(128, kernel_regularizer=regularizers.l2(l2_reg_weight),
                         bias_regularizer=regularizers.l2(l2_reg_weight), activation='relu'))
        
        # model.add(Dense(64, kernel_regularizer=regularizers.l2(l2_reg_weight),
        #                  bias_regularizer=regularizers.l2(l2_reg_weight),activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, kernel_regularizer=regularizers.l2(l2_reg_weight),
                         bias_regularizer=regularizers.l2(l2_reg_weight), activation='sigmoid'))
        
        # model.add(Conv2D(8, (3, 3), kernel_regularizer=regularizers.l2(l2_reg_weight),
        #                  bias_regularizer=regularizers.l2(l2_reg_weight),activation='relu',padding='same'))
        # model.add(Conv2D(4, (3, 3), kernel_regularizer=regularizers.l2(l2_reg_weight),
        #                  bias_regularizer=regularizers.l2(l2_reg_weight),activation='relu',padding='same'))
        
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        
        self.model  = model
        
        
    def generate(self):        
        #Set tainable attribute of [:freeze] to False
        #self.freeze_top_layers()
        
        #Set all layers to trainable as True
        if config.configured_trainable:        
            self.set_trainable(self.model, True)      #set all layer of self-defined as trainable

        self.model.compile(loss = config.configured_loss,
                            optimizer = self.optimizer, #optimizers.Adam(lr=0, beta_1=0.9, beta_2=0.999, decay=lr_decay),        #learning rate, default = 0.001
                            metrics = config.configured_metrics)
        
        #print the overview information of this model
        self.model.summary()
        
        #Change: save checkpoint
        self.callbacks = self.get_callbacks(config.get_fine_tuned_weights_path(True), 
                                       patience = self.train_patience)
       
    
    
    def train(self):
        
        self.dataset.debug()

        self.model.fit_generator(
            self.dataset.get_dataflow(config.configured_train_dir),
            steps_per_epoch = self.dataset.get_phase_count(config.configured_train_dir) / float(config.configured_batch_size),
            epochs = config.configured_epoch,
            validation_data = self.dataset.get_dataflow(config.configured_valid_dir),
            validation_steps = self.dataset.get_phase_count(config.configured_valid_dir) / float(config.configured_batch_size),
            callbacks = self.callbacks,
            class_weight = self.dataset.get_class_weight())

        self.model.save(model_manager.get_model_file_path())
        
        self.save_classes()
        
    def fineturn(self, keras_model):
        x = keras_model.output
        
        x = Flatten()(x)
        x = Dense(4096, activation='elu', name='fc1')(x)
        x = Dropout(0.6)(x)
        x = Dense(self.noveltyDetectionLayerSize, activation='elu', name=self.noveltyDetectionLayerName)(x)
        x = Dropout(0.6)(x)
        
        predictions = Dense(len(config.configured_classes), activation='softmax', name='predictions')(x)

        return Model(input = keras_model.input, output=predictions)


def inst_class(*args, **kwargs):
    print('args: ', args)
    print('keyargs: ', kwargs)
    return AlexNetSJTU(*args, **kwargs)
