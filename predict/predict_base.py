import time
import argparse
import os
import numpy as np
import glob
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

import util
import traceback
from keras.applications.densenet import decode_predictions
from dataset import dataset
from config import config
from sklearn.svm.libsvm import predict
from asyncio.tasks import sleep
from exception import C8Exception as c8e


class predict_base():
    def __init__(self, dt = None, model = None, 
                 output_file = config.configured_predict_output,
                 result_output_file = config.configured_predict_result, 
                 wrong_output_file = config.configured_predict_wrong):
        self.__dt               = dt
        if None != dt:
            self.__sample_count     = dt.get_phase_count(config.configured_predict_dir)
            self.__dir              = dt.get_dir_path(config.configured_predict_dir)
        self.__model            = model
        
        self.__fd_output        = open(output_file, 'w+')
        self.__fd_result        = open(result_output_file, 'w+')
        self.__fd_wrong         = open(wrong_output_file, 'w+')
        
    def get_sample_count(self):
        return self.__sample_count
    
    def get_predict_dir(self):
        return self.__dir
        
    def get_inputs_and_trues(self, image_path, label):
        """Internal API
        """
        inputs = []
        y_true = []
        
        img = image.load_img(image_path, target_size = config.configured_image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        
        x = preprocess_input(x)
    
        inputs.append(x)
        y_true.append(label)

        return y_true, inputs
                
    def predict_one_image(self, image_path, label = None):
        """Internal API
        """
        #print('Predict', image_path)  
        
        y_true, inputs = self.get_inputs_and_trues(image_path, label)
        out = self.__model.predict(inputs)
        
        #print('Out: ', out)
        
        #Check the max predict index
        index = np.argmax(out)
        
        msg = 'image: {}; label: {}, predict:{}\n'.format(image_path, str(label), out)
        
        
        if label:        
            if index == int(label):
                #print('correct predict for image: ', image_path, 'with label: ', label)
                self.__fd_result.write(msg)
                return [True, out, index]
            else:
                #print('wrong predict for image: ', image_path, 'with label: ', label)
                self.__fd_result.write(msg)
                self.__fd_wrong.write(msg)
                
                self.__fd_result.flush()
                self.__fd_wrong.flush()
                
                return [False, out, index]
        else:
            #print('predict for image: ', image_path, 'with label: ', label)
            self.__fd_result.write(msg)
            return [None, out, index]
            
        
        
    def predict_one_group(self, images, labels = None):
        """
        External API
        The parameter images should be list stored images path
        Input:
            images: the image(s) for predicting
            lables: optional, for check the predicting result
        Output:
            list: [result, outs, classes]
                results: comparation between predicting and given lables
                outs: the 2D list for each images
                    1st list is the probability of each image
                    2nd list is the probability of each class of the image
                classes: 
                    The actually predicting results, that is the class order
        """
        
        results = []
        outs = []
        classes = []
        
        if (None != labels) and (len(images) != len(labels)):
            raise c8e('Wrong parameters')
               
        for i in range(len(images)):
            result, out, index = self.predict_one_image(images[i], labels[i] if labels != None else None)
            results.append(result)
            outs.append(out)
            classes.append(index)
            
        if labels:
            return [results, outs, classes]
        else:
            return [None, outs, classes]
        
    def predict_one_group_for_BP(self, images, labels = None):
        """For BP
            Output is 1D array, consisting of probability of classes of each image
        """
        results, outs, classes = self.predict_one_group(images, labels)
        
        outs1d = []
        for i in range(len(outs)):
            for j in range(len(outs[0])):
                outs1d.append(outs[i][j])
                
        return outs1d
        
if __name__ == '__main__':
    print('test')        