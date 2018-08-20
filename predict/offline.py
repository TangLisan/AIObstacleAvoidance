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
import dataset.dataset_group
from dataset.dataset_group import dataset_group


class preoffline(predict_base):
    def __init__(self, dt = None, model = None):
        predict_base.__init__(self, dt, model)
        
        
    def predict_all_images(self):
        """Internal API
        """
        print('Sample information count: ', self.get_sample_count())
        
        predict_true = 0
        total_images = 0
        
        results = []
        outs = []
        
        for l1 in os.listdir(self.get_predict_dir()):
            
            l1path = os.path.join(self.get_predict_dir(), l1)
            for l2 in os.listdir(l1path):
                total_images += 1 
                l2path = os.path.join(l1path, l2)
                
                result, out, index = self.predict_one_image(l2path, l1)
                results.append(result)
                outs.append(out)
                
                if True == result:
                    predict_true += 1
        
        print('Predict accuracy: ', (1.0 * predict_true) / total_images)
        
        return results, outs
        
    def predict_groups(self):
        """Used in the situation, that only giving 3 folders
        """
        print('Not support now')
        
        
    def predict_groups_with_folder(self):
        """Predict all images with group form, that is, this function should find all suitable 
        groups from the given folder        
        """
        print('Sample information count: ', self.get_sample_count())
        
        predict_true = 0
        total_images = 0
        
        results_total = []
        outs_total = []
        
        predict_true_group = 0
        total_groups = 0
        
        #find all groups in predict directory
        dg = dataset_group(self.get_predict_dir())
        groups = dg.find()
           
        for k, v in groups.items():
            print('class: ', k)
            for group in groups[k]:
                print('images in one group:',
                      group[0],
                      group[1],
                      group[2])                
                
                #predict current group
                results, outs, classes = self.predict_one_group(group, [k, k, k])
                
                group_flag = True  #identify accuracy of one group
                for i in range(len(results)):                
                    if True == results[i]:
                        predict_true += 1
                    else:
                        group_flag = False    
                
                if True == group_flag:
                    predict_true_group += 1
                
                total_images += 3   
                total_groups += 1 
                
                outs_total.append(outs)
                results_total.append(results)
        
        print('==' * 50)
        print('Predict accuracy with single image form: ', (1.0 * predict_true) / total_images)
        print('Predict accuracy with group  form: ', (1.0 * predict_true_group) / total_groups)
        
        return results_total, outs_total
        
        
if __name__ == '__main__':
    print('test')        