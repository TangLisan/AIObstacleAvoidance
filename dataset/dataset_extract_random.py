"""
This file is used to rename image files from sjtu and classs them into keras orgnization structure
"""

import os
import os.path 
import shutil 
import random
from config import config

from exception import C8Exception as c8e

from .dataset_base import dataset_base



class dataset_extract_random(dataset_base):
    def __init__(self, inpath, outpath):
        self.__inpath   = inpath
        self.__outpath  = outpath

        self.__extract_fd = open(config.configured_extract_output_file, 'w+')
        
        dataset_base.__init__(self)
        
        
    def checkenv(self):
        """
        Check whether input directory exist
        Create directory and sub-directory for output path
        """
        if not os.path.exists(self.__inpath):
            raise c8e('Input directory not exist {}'.format(self.__inpath))
            
        if os.path.exists(self.__outpath):
            shutil.rmtree(self.__outpath)
            
        os.mkdir(self.__outpath)
        
        for phase in config.configured_dirs:
            phase_path = os.path.join(self.__outpath, phase)
            os.mkdir(phase_path)
        
            for label in config.configured_classes:
                subpath = os.path.join(phase_path, label)
                os.mkdir(subpath)
        
        #Need check again?
    
    def is_camera0_image(self, image):
        
        slices = image.split('-')
        if '0' == slices[1]:    #2nd field in image name
            return True
        else:
            return False
    
    
    def find_images_camera0(self, classpath):   
        
        images = []
        for it in os.listdir(classpath):
            if self.is_camera0_image(it):
                images.append(it)
        
        return images
    
    def extract_random_process(self, images_camera0, classes, phase, count):
        
        for i in range(count):
            if 0 >= len(images_camera0):
                return
            
            image = random.choice(images_camera0)
            
            group = self.find_related_file(os.path.join(self.__inpath, classes), 
                                           image)
            
            for it in group:
                head, tail = os.path.split(it)  
                
                abs_dst_file = os.path.join(self.__outpath,
                                            phase,
                                            classes,
                                            tail)
                shutil.copy(it, abs_dst_file)
                print('copy from', it, 'to', abs_dst_file)  
            
                self.__extract_fd.write('{} -> {}'.format(it, abs_dst_file))
                
            images_camera0.remove(image)


    def extract_random(self, images_camera0, classes):
        
        total = len(images_camera0) 
        
        valid_count = int(total * config.configured_phase_percent[1])   #index of valid
        predict_count = int(total * config.configured_phase_percent[2]) #index of predict
        train_count = total - valid_count - predict_count
        
        self.extract_random_process(images_camera0, classes, config.configured_valid_dir, valid_count)
        self.extract_random_process(images_camera0, classes, config.configured_predict_dir, predict_count)
        self.extract_random_process(images_camera0, classes, config.configured_train_dir, train_count)
        
        
    def extract(self):
        """Extract random group images into specified folder
        """
        
        self.checkenv()
        
        for l1 in os.listdir(self.__inpath):
            
            if l1 not in config.configured_classes:
                raise c8e('wrong folder in input directory: {}, its {}'.format(self.__inpath, l1))
            
            l1path = os.path.join(self.__inpath, l1)
            images_camera0 = self.find_images_camera0(l1path)
            if 3 >= len(images_camera0):
                raise c8e('Wrong images count in {}'.format(l1path)) 
            
            self.extract_random(images_camera0, l1) #l1 is the classes name
          
                
        