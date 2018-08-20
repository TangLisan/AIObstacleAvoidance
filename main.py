
from train import train
from predict import predict
from dataset.dataset_extract_random import dataset_extract_random
from dataset.dataset_rename import dataset_rename
from config import config
from config import env
from exception import C8Exception
import traceback
from models import model_weight as mw
import os
# from keras_applications import vgg16
# from keras_applications import get_keras_submodule
# from keras.applications.vgg16 import VGG16 as KerasVGG16 


def check_weight_file():
    if os.path.exists(config.configured_pretrained_weight):
        mw.print_keras_wegiths(config.configured_pretrained_weight)

def dataset_preprocess():
    rn = dataset_rename(config.configured_input_image_precessing_dir,
                          config.configured_output_rename_dir)
    
    rn.process()
    
    et = dataset_extract_random(config.configured_output_rename_dir,
                                config.configured_output_extract_dir)
    et.extract()    
      
          

if __name__ == '__main__':
    try:
    
        #check_weight_file()    
        env.check_env()
        
        #if need preprocess dataset
        #dataset_preprocess()
        
        train.train()
        
        predict.predict_direct(config.configured_predict_modes[1])
        
    except Exception as e:
        print(e)
        traceback.print_exc()
    except C8Exception as c8e:
        print(c8e)
        traceback.print_exc()
    finally:
        print('over')
