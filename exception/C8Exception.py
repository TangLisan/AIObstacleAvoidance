


class C8Exception(Exception):
    def __init__(self, msg):
        self.msg = msg
        
    def print(self):
        print('C8 Exception', self.msg)
        
        