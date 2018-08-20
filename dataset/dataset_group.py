"""
This file is used to find the group in specific folder
"""

import os
import os.path 
import shutil 
import random
from config import config
from .dataset_base import dataset_base


class dataset_group(dataset_base):
    def __init__(self, folder = None):
        self.__folder   = folder
        dataset_base.__init__(self)
        
        
    def find(self):
        """The return value format:
        {class: [[image1_of_group1, image2_of_group1, image3_of_group1], [ , , ], ...], ...}
        The finding rule is: 
            1. 1st field should be corresponding with folder name
            2. 2nd field should be the index, we only tranverse the '0', 
            3. construct the other two images, using the 1st, 2nd and 3rd fields of the index image
        e.g. 
             traverse one image file named: R-0-39.jpg, then we get the other two images:
                 R-1-39.jpg
                 R-2-39.jpg
        """
        
        if not self.__folder:
            print('Wrong parameters')
            return dict()
        
        #initialize the return value
        retval = dict()
        
        for l1 in os.listdir(self.__folder):    #classes folder, 0/1/2
            
            groups = []
            
            l1path = os.path.join(self.__folder, l1)
            if not os.path.isdir(l1path):
                print('Wrong file found')
                continue
            
            for l2 in os.listdir(l1path):    #image files
                l2path = os.path.join(l1path, l2)
                if os.path.isdir(l2path):
                    print('Wrong folder fould, should be image file', l2path)
                    continue
                
                #it's image from camera 0
                slices = l2.split('-')
                if '0' != slices[1]:
                    continue
                
                group = self.find_related_file(l1path, l2)
                if 0 == len(group):
                    print('One time wrong finding operation')
                    continue
                
                groups.append(group)
                
            retval[l1] = groups    
        
        return retval
        

if __name__ == '__main__':
    test = dataset_group(os.path.join(config.configured_root_path, config.configured_predict_dir))
    
    retval = test.find()
    
    for k, v in retval.items():
        print('class: ', k)
        for group in retval[k]:
            print('images in one group:',
                  group[0],
                  group[1],
                  group[2])
        
        
        
        
        
