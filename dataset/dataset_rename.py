"""
This file is used to rename image files from sjtu and classs them into keras orgnization structure
"""

import os
import os.path 
import shutil 
import random
from config import config

"""For data set from sjtu, we need to rename all file name currently.
The naming rule of new image file is dir1_dir2_filename
input_path directory structure need bo be:
 input
   |--camera0
   |   |--0
   |   |    |--XXX.jpg
output_path directory structure need bo be:
 output(this directory need bo be created manually, the sub-directory(train/valid/test) will be create automatically in this py file)
   |--0 #Red
   |   |  |--R_0_AAA.jpg    #camera 0
   |   |  |--R_1_AAA.jpg    #camera 1
   |   |  |--R_2_AAA.jpg    #camera 2
   |--1 #Yellow
   |   |  |--Y_0_BBB.jpg    #camera 0
   |   |  |--Y_1_BBB.jpg    #camera 1
   |   |  |--Y_2_BBB.jpg    #camera 2
   |--2 #Green
   |   |  |--G_0_CCC.jpg    #camera 0
   |   |  |--G_1_CCC.jpg    #camera 1
   |   |  |--G_2_CCC.jpg    #camera 2

sjtu change image file rule, thus change...


"""



class dataset_rename():
    def __init__(self, inpath, outpath):
        self.__inpath = inpath
        self.__outpath = outpath
        self.__rename_fd = open(config.configured_rename_output_file, 'w+')


    def checkenv(self):
        """Check the input directory and out directory
        Create classes level directory for output
        Output: False: failed
                True: success
        """
        if not os.path.isdir(self.__inpath):
            print('Input folder not exist, exit', self.__inpath)
            return False
            
        #clear all older files for avoiding mistakes    
        if os.path.exists(self.__outpath):
            shutil.rmtree(self.__outpath)
        
        os.mkdir(self.__outpath)
            
        #create class directory: 0/1/2    
        for it in config.configured_classes:
            os.mkdir(os.path.join(self.__outpath, it))
            
        glb_check_flag = True        
        return glb_check_flag
    
    def get_dirname(self, path):
        """Find all 1st level directory from given folder 
        Return:
            list consisting with 1st level directory name
            empty list if failed
        """
        dirs = []
        
        if False == os.path.isdir(path):
            return dirs
        
        for direct in os.listdir(path):
            absdir = os.path.join(path, direct)
            if False == os.path.isdir(absdir):
                continue
            
            dirs.append([absdir, direct])
            
        return dirs
    
    def format_file(self, l1, l2, file):
        """Reanme one file following the naming rules we defined
        """
        
        #if this flag enabled, process the image file like dataset16_25.jpg
        #rename it to 25.jpg
        if config.configured_dataset_new:
            slices = file.split('_')
            file = slices[1]        
    
        newfile = ''
        
        if '0' == l2:
            newfile += 'G'
        elif '1' == l2:
            newfile += 'Y'
        elif '2' == l2:
            newfile += 'R'
        
        newfile += '-'
        
        if 'camera0' == l1:
            newfile += '0'
        elif  'camera1' == l1:
            newfile += '1'
        elif  'camera2' == l1:
            newfile += '2'
            
        newfile += '-'        
        newfile += file
              
        return newfile
  
    
    def move_files(self, files, classes):
        """
        Copy images of one class of one camera with new names to destinate directory
        """        
        for k, v in files.items():
            abs_dst_filename = os.path.join(self.__outpath, classes, k)
            shutil.copy(v, abs_dst_filename)
            
        return True
    
    def process(self):
        
        if False == self.checkenv():
            return False
        
        
        #find all image file and rename them and store into a list with area name
        #The 1st level folder should be camera0/1/2
        for l1 in self.get_dirname(self.__inpath):
            
            for l2 in self.get_dirname(l1[0]):    #2nd level folder should be 0/1/2, the classes
                
                dst_files = {}
            
                for file in os.listdir(l2[0]): #3rd should be images
                    
                    abs_src_file = os.path.join(l2[0], file)
                    if True == os.path.isdir(abs_src_file):
                        print('Not file', abs_src_file)
                        continue
                    
                    dst_file = self.format_file(l1[1], l2[1], file)
                    print('src file', abs_src_file, '\n', 'dst file', dst_file, int(l2[1]))

                    dst_files[dst_file] = abs_src_file
            
                #move the specified count images with random order to dest folder
                self.move_files(dst_files, l2[1]) #l2[1] is the class of images
        
        return True
    

if __name__ == '__main__':
    test = dataset_rename(config.configured_input_image_precessing_dir,
                          config.configured_output_rename_dir)
    
    test.process()

