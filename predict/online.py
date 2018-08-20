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
from dataset import dataset
from config import config
from sklearn.svm.libsvm import predict
from asyncio.tasks import sleep
from predict.predict_base import predict_base


class preonline(predict_base):
    def __init__(self, dt = None, model = None):
        predict_base.__init__(self, dt, model)
    
    def predict_with_group(self, images, labels = None):
        """      
        """
        inputs = []
        inputs.append(images['images'][0]['image'])
        inputs.append(images['images'][1]['image'])
        inputs.append(images['images'][2]['image'])

        
        outs = []
        labels = []
        
        for i in range(3):
            result, out, label = self.predict_one_image(inputs[i])
            outs.append(out)
            labels.append(label)
        
        #predict current group
        #results, outs, labels = self.predict_one_group(inputs)
        
        return [outs, labels]
        
        
if __name__ == '__main__':
    print('test')        