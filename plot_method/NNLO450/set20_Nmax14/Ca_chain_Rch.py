import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt 

import re

from scipy import interpolate
from math import log 
from math import e


def input_file_2(file_path,raw_data):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0 
        loop1 = 0 
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                raw_data[loop2][0] = float(temp_1[0])   # A
                raw_data[loop2][1] = float(temp_1[2])   # deltannlo_new
                raw_data[loop2][2] = float(temp_1[3])   # exp
                raw_data[loop2][3] = float(temp_1[4])   # nnlosat
                raw_data[loop2][4] = float(temp_1[5])   # magic
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 


file_path    = "Ca_chain_Rch.txt"
data_num     =  4
Ca_chain_r_data = np.zeros((data_num,5),dtype = np.float)
input_file_2(file_path,Ca_chain_r_data)


#start plotting
fig_1 = plt.figure('fig_1')
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
plt.tick_params(top=True,bottom=True,left=True,right=False)
#start_line = 40

x_list_exp      = Ca_chain_r_data[0:3,0]  # A of the O isotopes
y_list_exp      = Ca_chain_r_data[0:3,2]  # experiment binding energy

x_list_DNNLO450 = Ca_chain_r_data[:,0]  # A of the O isotopes
y_list_DNNLO450 = Ca_chain_r_data[:,1]  # delta nnlo 450 calculation

x_list_NNLOsat  = Ca_chain_r_data[:,0]  # A of the O isotopes
y_list_NNLOsat  = Ca_chain_r_data[:,3]  # nnlosat calculation

x_list_magic = Ca_chain_r_data[:,0]  # A of the O isotopes
y_list_magic = Ca_chain_r_data[:,4]  # magic 1.8/2.0 calculation


l_exp         = plt.scatter(x_list_exp,y_list_exp,color = 'k',s=50,marker = 'o',label='Expt.')
l_DNNLO450    = plt.scatter(x_list_DNNLO450,y_list_DNNLO450,color = 'b',s=50,marker = '^',label='DNNLO450$_{new}$')
l_NNLOsat     = plt.scatter(x_list_NNLOsat,y_list_NNLOsat,color = 'r',s=50,marker = 'D',label='NNLO$_{sat}$')
l_magic       = plt.scatter(x_list_magic,y_list_magic,color = 'c',s=50,marker = 'p',label='1.8/2.0')

plt.ylabel(r'$R_{\rm{ch}}$[fm]',fontsize=18)
plt.xlabel('$A$',fontsize=18)
plt.yticks(np.arange(3.25,3.7,0.1),fontsize = 13)
plt.xticks(np.arange(40,55,2),fontsize = 14)
#plt.legend(loc=2, bbox_to_anchor=(1.63,0.5),borderaxespad = 0.)
plt.legend(loc=2,bbox_to_anchor=(0.03,0.97))

plt.title('Charge radii of calcium isotopes')
plot_path = 'Ca_chain_Rch.eps'
plt.savefig(plot_path)
plt.show()
plt.close("all")


