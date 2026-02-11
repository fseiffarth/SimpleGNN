'''
Created on 07.01.2020

@author:
'''

import time

class TimeClass(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.time_dict = {}
        
    
    def measure(self, flag):
        if flag in self.time_dict:
            self.time_dict[flag].append(time.time())
        else:
            self.time_dict[flag] = [time.time()]
            
    def print_times(self):
        for x in self.time_dict.keys():
            if len(self.time_dict[x]) > 1:
                time_sum = 0
                for i, j in zip(self.time_dict[x][0::2], self.time_dict[x][1::2]):
                    time_sum += j-i
                print("Flag: {} time {} s".format(x, str(time_sum)))
                self.time_dict[x] = []
            
    def print_times_flag(self, flag):
        if flag in self.time_dict.keys():
            time_sum = 0
            for i, j in zip(self.time_dict[flag][0::2], self.time_dict[flag][1::2]):
                time_sum += j-i
            print("Flag: {} time {} s".format(flag, str(time_sum)))
            self.time_dict[flag] = []
           
    def get_flag_time(self, flag):
        if flag in self.time_dict.keys():
            time_sum = 0
            for i, j in zip(self.time_dict[flag][0::2], self.time_dict[flag][1::2]):
                time_sum += j-i
            self.time_dict[flag] = []
            return time_sum

    def reset(self):
        for x in self.time_dict.keys():
            if len(self.time_dict[x]) > 1:
                self.time_dict[x] = []
        