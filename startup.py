

import threading  
import socket  
import time
import json
import traceback


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
import tensorflow as tf


from config import config
from pprint import pprint



graph = []
graph.append(tf.get_default_graph())
  
encoding = 'utf-8'  
BUFSIZE = 1024  
  
# a read thread, read data from remote  
class PredictThread(threading.Thread):  
    def __init__(self, client, pre):  
        threading.Thread.__init__(self)
        self.pre = pre  
        self.client = client  
        self.__output_fd = open(config.configured_onlinepredict_output_file, 'w+')
    
    
    def decode(self, data):
        """
        This function will decode the json message to list
        The returned value type is dict
        """
        
        print('Message decode')
        try:
            images = json.loads(data.decode('utf-8'))
            
            pprint(images)
            
            return images
            
        except Exception as e:
            print(e)
            traceback.print_exc()
        finally:
            print('over')
            
    def encode(self, outs, classes):
        print('encoded')
        
        probs = []
        i = 0
        for l1 in outs:
            probitem = dict()
            probitem['probility'] = '{};{};{}'.format(l1[0][0], l1[0][1], l1[0][2])
            probitem['label'] = str(classes[i])
            i += 1
            
            probs.append(probitem)
        
        data = dict()
        data['predicts'] = probs;
    
        message =  json.dumps(data)
        print('encoded json: ', message)
        return message  
          
          
    def run(self):  
      
        print('enter thread to process for: ' + str(self.client))
        index = 0
        
        while True:
            data = self.client.recv(config.configured_msg_buffer_size)  
            print(data)
            
            self.__output_fd.write('\n\n----------------->{}\n'.format(index))
            self.__output_fd.write('Recv: {}\n'.format(data))
            
            #parse the received message to structure
            images = self.decode(data)
            
            with graph[0].as_default(): #for multi-thread environment
                #execute predict
                outs, classes = self.pre.execute(images)
            
            message = self.encode(outs, classes)
            
            self.client.send(message.encode(encoding='utf_8'))
            
            self.__output_fd.write('Send: {}\n'.format(message))
            
            index += 1
  
class SCServer(threading.Thread):  
    """
    The server main thread for monitoring connection and creating a thread to establish communication
    """
    def __init__(self, port, pre):  
        threading.Thread.__init__(self)  
        self.pre = pre
        self.port = port  
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  
        self.sock.bind(("0.0.0.0", port))  
        self.sock.listen(0)  
        
    def run(self):  
        print( "listener started"  )
        while True:  
            client, cltadd = self.sock.accept()  
            print ("accept a connect..."  )
            PredictThread(client, self.pre).start()
            cltadd = cltadd  *
            print( "accept a connect(new reader..)"  )
            
            
def predict_online_group(images):
    """
    Input:
        dict type images path
    """
    print('predict....')
    


def predict_startup():
    """
    this function will be invoked by AI SW with CPP
    The target of this functoin is used to, including
        1. Initialize the Python environment
        2. Load trained weight file
        3. Completed others, including creating debug files, etc.
    """
            
    #check environment
    env.config_env()
        
    pre = predict.predict()
    pre.load_model()
        
    
    scserv  = SCServer(config.configured_TCP_server_port, pre)
      
    scserv.start() 
 
 
if __name__ == '__main__':    
    """
    Used for unit test 
    """
    predict_startup()
    
        