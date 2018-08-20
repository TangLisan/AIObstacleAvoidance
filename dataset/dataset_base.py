"""
This file is used to find the group in specific folder
"""

import os
import os.path 
import shutil 
import random
from config import config

class dataset_base():
    def __init__(self):
        print('base init')
    
    
    def find_related_file(self, parent_path, image):
        
        slices = image.split('-')
                
        #image from camera 1
        c1_image = slices[0] + '-1-' + slices[2]
        c1_image_path = os.path.join(parent_path, c1_image)        
        if not os.path.exists(c1_image_path):
            print('Cannot find corresponding image for', os.path.join(parent_path, image), c1_image_path)
            return []
        
        #image from camera 2
        c2_image = slices[0] + '-2-' + slices[2]
        c2_image_path = os.path.join(parent_path, c2_image)        
        if not os.path.exists(c2_image_path):
            print('Cannot find corresponding image for', os.path.join(parent_path, image), c2_image_path)
            return []
        
        #add
        return [os.path.join(parent_path, image),
                c1_image_path,
                c2_image_path]
        