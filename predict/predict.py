import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib

import util
import traceback
from keras.applications.densenet import decode_predictions
from predict import online


"""
Use Default parameters value
"""

from models import model_base
from dataset import dataset
from models import model_manager
from config import config
from config import env
from predict.offline import preoffline as pof
from predict.online import preonline as pon


class predict():
    def __init__(self, mode = config.configured_predict_modes[0]):
        self.__mode = mode
        self.__loaded_model = None  
        
        self.__dataset = dataset.dataset()  
           
    def load_model(self):  
        
        #instance and initialize  model
        model = model_manager.get_model_instance(dataset = self.__dataset)      
        model.define()
        self.__loaded_model = model.load_trained_model()
        
        if self.__mode == config.configured_predict_modes[0]:
            print('Online not support')
            self.__pon = pon(None, self.__loaded_model)            
        else:     
        
            #check dataset
            self.__dataset.check()
            self.__dataset.generate_dataflow(image_size = config.configured_image_size)
            
            self.__dataset.debug()  #for debuging data set   
            self.__pof = pof(self.__dataset, self.__loaded_model)
        
    def execute(self, images = None):
        """
        Input:
            images: dict
        """

        tic = time.clock()
        
        if self.__mode == config.configured_predict_modes[0]:
            return self.__pon.predict_with_group(images)
        elif self.__mode == config.configured_predict_modes[1]:        
            self.__pof.predict_all_images() 
                   
        elif self.__mode == config.configured_predict_modes[2]:       
            self.__pof.predict_groups_with_folder()
            
        elif self.__mode == config.configured_predict_modes[3]:     
            print('Offline with input with group form not support')                      
        
        toc = time.clock()
        print('Time: %s' % (toc - tic))    
        
        
def predict_direct(mode):
    """Entry point of predict    
    """
    pre = predict(config.configured_predict_modes[1])
    pre.load_model()
    pre.execute()
    
        
if __name__ == '__main__':
    predict_direct()
        
