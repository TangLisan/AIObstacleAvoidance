from os.path import join as join_path
import os

from config import config_dataset
from config import config_platform
from config import config_model

"""
This configuration is used to confirm whether using the standard model of Keras
If False, using the standard model
If True, using fineturning model
"""
use_fineturn_model = False


"""
IMPORTANT for data set
if use data set >= 16, such as dataset16.7z, ENABLE this flag
"""
configured_dataset_new      = True




###############################################################################
#Data set processing related config

configured_use_vrep = True  

#root path
configured_root_dir                         = 'D:/20_Workspace/02_AI/02_data_set/dataset15_2_224'

#input for renaming image files
configured_input_image_precessing_dir       = configured_root_dir + 'dataset16'

#output for renaming image files, input for extracting image files
configured_output_rename_dir                = configured_root_dir + 'dataset15_rename'

#output for extracting image files
configured_output_extract_dir               = configured_root_dir + 'dataset15_extract'

#input for training
configured_root_path                        = configured_root_dir

#directory for storing trained modelor weight file
configured_trained_dir                      = './trained/'

#directory to store CNN model
configured_model_path                       = './models'

#for storing 
configured_debug_dir                        = './debug/'
configured_train_result                     = configured_debug_dir + 'train_results.txt'
configured_predict_result                   = configured_debug_dir + 'predict_results.txt'
configured_predict_wrong                    = configured_debug_dir + 'predict_wrong_result.txt'
configured_predict_output                   = configured_debug_dir + 'predict_output.txt'
configured_rename_output_file               = configured_debug_dir + 'dataset_rename_output_file.txt'
configured_extract_output_file              = configured_debug_dir + 'dataset_extract_output_file.txt'

configured_onlinepredict_output_file        = configured_debug_dir + 'online_predict_output_file.txt'

configured_pretrained_weight                = 'D:/20_Workspace/02_AI/01_keras_pretrained_weight_files/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


"""Global variables used in this module
"""
glb_check_flag = False

#For extract image files from given data set
#the list item is the percent of file total count
configured_phase_percent = [0.70, 0.15, 0.15]  #for train, valid and predict


###############################################################################
#data set configuration


#folder name for different phase
configured_train_dir        = 'train'
configured_valid_dir        = 'valid'
configured_predict_dir      = 'predict'
configured_dirs             = [configured_train_dir, configured_valid_dir, configured_predict_dir]

#area name, also act as folder name
configured_red_class        = '2'
configured_yellow_class     = '1'
configured_green_class      = '0'

#label information
configured_classes          = [configured_green_class, configured_yellow_class, configured_red_class]
configured_classes_count    = len(configured_classes)

#sample image format
configured_image_format     = ['png', 'jpg', 'jpeg', 'bmp', 'JPG']

###############################################################################
#model configuration 

MODEL_VGG16             = 'vgg16'
MODEL_INCEPTION_V3      = 'inception_v3'
MODEL_INCEPTION_V4      = 'inception_v4'
MODEL_RESNET50          = 'resnet50'
MODEL_RESNET152         = 'resnet152'
MODEL_ONNX              = 'onnx'
MODEL_SENET             = 'senet'
MODEL_ALEXNET           = 'alexnet'
MODEL_ALEXNET_SJTU      = 'alexnet_sjtu'
MODEL_ALEXNET_1         = 'alexnet_1'

"""
Define the model related parameters
"""

configured_model            = MODEL_VGG16
configured_batch_size       = 800
configured_epoch            = 400
configured_image_size       = (224, 224)
configured_freeze_layers    = 0
configured_train_patience   = 20
configured_weight           = None
configured_learning_rate    = 1e-5
configured_trainable        = True
configured_loss             = 'categorical_crossentropy'
configured_metrics          = ['accuracy']

configured_predict_modes    = ['online', 'offline_single', 'offline_group_folder', 'offline_group_para']

#trained weight file name
configured_date             = ''
configured_model_weights                    = configured_trained_dir + configured_date +configured_model +'-trained_weight.h5'  #weight file of trained model


###############################################################################
#Configuration with UI and other CPP

configured_TCP_server_port      = 8888

configured_msg_buffer_size      = 1024

###############################################################################
#For debug
configured_model_json_file          = './debug/model_json.json'




abspath = os.path.dirname(os.path.abspath(__file__))

lock_file = os.path.join(abspath, 'lock')

data_dir = join_path(abspath, 'data/sorted')
trained_dir = join_path(abspath, 'trained')

train_dir, validation_dir = None, None

MODEL_VGG16 = 'vgg16'
MODEL_INCEPTION_V3 = 'inception_v3'
MODEL_RESNET50 = 'resnet50'
MODEL_RESNET152 = 'resnet152'
MODEL_ALEXNET = 'alexnet'

model = MODEL_RESNET50

bf_train_path = join_path(trained_dir, 'bottleneck_features_train.npy')
bf_valid_path = join_path(trained_dir, 'bottleneck_features_validation.npy')
top_model_weights_path = join_path(trained_dir, 'top-model-{}-weights.h5')
fine_tuned_weights_path = join_path(trained_dir, 'fine-tuned-{}-weights.h5')
model_path = join_path(trained_dir, 'model-{}.h5')
classes_path = join_path(trained_dir, 'classes-{}')

activations_path = join_path(trained_dir, 'activations.csv')
novelty_detection_model_path = join_path(trained_dir, 'novelty_detection-model-{}')

plots_dir = join_path(abspath, 'plots')

# server settings
server_address = ('0.0.0.0', 4224)
buffer_size = 4096


nb_train_samples = 0
nb_validation_samples = 0


def set_paths():
    global train_dir, validation_dir
    train_dir = join_path(data_dir, 'train/')
    validation_dir = join_path(data_dir, 'valid/')


set_paths()


def get_top_model_weights_path():
    return top_model_weights_path.format(model)


def get_fine_tuned_weights_path(checkpoint=False):
    return fine_tuned_weights_path.format(model + '-checkpoint' if checkpoint else model)


def get_novelty_detection_model_path():
    return novelty_detection_model_path.format(model)


def get_model_path():
    return model_path.format(model)


def get_classes_path():
    return classes_path.format(model)
