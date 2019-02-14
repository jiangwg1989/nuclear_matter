import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt 

import re

from scipy import interpolate
from math import log 
from math import e


def input_file_2(file_path,raw_data):
    count = len(raw_data)
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0 
        loop1 = 0 
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                raw_data[loop2][0] = float(temp_1[2])   # exp
                raw_data[loop2][1] = float(temp_1[3])   # cal
#                raw_data[loop2][2] = float(temp_1[4])
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 


file_path    = "O_chain_.txt"
data_num     =  4
O_chain_data = np.zeros((data_num,2),dtype = np.float)
input_file_2(file_path,O_chain_data)


print('O_exp='+str())
