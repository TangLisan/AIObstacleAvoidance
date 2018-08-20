from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.optimizers import Adam
import numpy as np
from sklearn.externals import joblib
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
from exception import C8Exception as c8e

import util
from asyncio.tasks import sleep

class dataset():
    def __init__(self):        
        self.__dataflow                     = dict()        
        self.__path                         = dict()
        
        self.__phase_sample_count           = {config.configured_train_dir : 0,
                                               config.configured_valid_dir : 0,
                                               config.configured_predict_dir : 0}
        
        self.__train_sample_count           = {config.configured_green_class : 0,
                                               config.configured_yellow_class : 0,
                                               config.configured_red_class : 0}
        
        self.__area_sample_count            = {config.configured_green_class : 0,
                                               config.configured_yellow_class : 0,
                                               config.configured_red_class : 0}
        
        self.__class_weight                 = {0 : 0.0,
                                               1 : 0.0,
                                               2 : 0.0}
        
    def get_phase_count(self, phase = None):
        if phase == None:
            total = 0
            for k, v in self.__phase_sample_count.items():
                total += v
            return total
        
        if phase not in config.configured_dirs:
            raise c8e('wrong phase name')
        
        return self.__phase_sample_count[phase]
    
    def get_area_count(self, area = None):
        if area == None:
            total = 0
            for k, v in self.__area_sample_count.items():
                total += v
            return total
        
        if area not in config.configured_classes:
            raise c8e('wrong area name')
        
        return self.__area_sample_count[area]
    
    def get_train_sample_count(self):
        total = 0
        for k, v in self.__train_sample_count.items():
            total += v
        return total
    
        
    def _check_image(self, l2path, file):
        _, ext = os.path.splitext(file)        
        if ext[1:] not in config.configured_image_format:   #valid file
            raise c8e('wrong imamge format' + l2path + file)
        
        return True
        
        
    def check(self):
        """Check the specified samples, including
            1. check the directory name of configured phase
            2. check the directory name of configured area/class
            3. check the image format of each image
            4. statistic image information
            5. compute the weight of each class of the samples of train phase
        """
        if not os.path.exists(config.configured_root_path):
            raise c8e('root path of data set not exist' + config.configured_root_path)
        
        for it in config.configured_dirs:
            self.__path[it] = os.path.join(config.configured_root_path, it)
        
        for l1 in os.listdir(config.configured_root_path):  #1st level directory: train/valid/test
            l1path = os.path.join(config.configured_root_path, l1)
            if not os.path.isdir(l1path):
                raise c8e('regular file exists in data set' + l1)
                        
            if l1 not in config.configured_dirs:
                raise c8e('wrong dir in data set' + l1)
            
            for l2 in os.listdir(l1path):   #2nd level directory, red/yellow/green
                l2path = os.path.join(l1path, l2)
                if not os.path.isdir(l2path):
                    raise c8e('regular file exists in data set' + l2path)
                            
                if l2 not in config.configured_classes:
                    raise c8e('wrong dir in data set' + l2path)
                
                for l3 in os.listdir(l2path):      #3rd level files          
                    self._check_image(l2path, l3)
                
                    #statistic
                    self.__phase_sample_count[l1] += 1
                    self.__area_sample_count[l2] += 1
                    if l1 == config.configured_train_dir:
                        self.__train_sample_count[l2] += 1
                    
        #process weight of each class of train directory
        total = np.sum(list(self.__train_sample_count.values()))   
        max_samples = np.max(list(self.__train_sample_count.values()))   #Max
        mu = 1. / (total / float(max_samples))          #
        keys = self.__train_sample_count.keys()
        for key in keys:
            score = math.log(mu * total / float(self.__train_sample_count[key]))
            self.__class_weight[int(key)] = score if score > 1. else 1.
                
    def get_class_weight(self):
        return self.__class_weight

    def generate_dataflow(self, image_size = (224, 224)):
        
        for it in config.configured_dirs:      
            idg = ImageDataGenerator()
        
            #Remove the mean operation, just using original images
            idg.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))
        
            self.__dataflow[it] = idg.flow_from_directory(
                                        self.get_dir_path(it), 
                                        target_size = image_size, 
                                        classes = config.configured_classes)

    def get_dataflow(self, phase = None):
        if None == phase:
            raise c8e('wrong phase')
        
        return self.__dataflow[phase]
    
    def override_keras_directory_iterator_next(self):
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
    
    
    
    def get_dir_path(self, phase = None):
        if None == phase:
            raise c8e('wrong pahse')
        
        return self.__path[phase]    

    def load_image(self, image_path, image_size):
        img = image.load_img(image_path, target_size = image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        
        return preprocess_input(x)
    
    def debug(self):
        """Print all information of data set
        """
        
        #path
        print('Path information:')
        for k, v in self.__path.items():
            print(k, v)
            
        #sample count
        print('Sample statistic of each phase')
        for k, v in self.__phase_sample_count.items():
            print(k, v)
            
        print('Sample statistic of each class')
        for k, v in self.__area_sample_count.items():
            print(k, v)
            
        print('Sample statistic of each train')
        for k, v in self.__train_sample_count.items():
            print(k, v)
    
if __name__ == '__main__':
    print('data set class test', '->' * 10)    
    
    dt = dataset()
    dt.check()
    
    dt.generate_dataflow()
    
    dt.debug()
    
    
    
    
    
    
    