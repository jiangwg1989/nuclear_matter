import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt 

import re

from scipy import interpolate
from math import log 
from math import e

def input_file_1(file_path,raw_data):
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
                raw_data[loop2][1] = float(temp_1[1])   # exp
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 


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
                raw_data[loop2][1] = float(temp_1[1])   # deltannlo450
                raw_data[loop2][2] = float(temp_1[2])   # exp
                raw_data[loop2][3] = float(temp_1[3])   # nnlosat
                raw_data[loop2][4] = float(temp_1[4])   # magic
                raw_data[loop2][5] = float(temp_1[5])   # deltannlo394
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 


file_path    = "Ca_chain_Rch.txt"
data_num     =  4
Ca_chain_r_data = np.zeros((data_num,6),dtype = np.float)
input_file_2(file_path,Ca_chain_r_data)

file_path    = "Ca_chain_Rch_exp.txt"
data_num     =  13 
Ca_chain_r_exp  = np.zeros((data_num,2),dtype = np.float)
input_file_1(file_path,Ca_chain_r_exp)
print(Ca_chain_r_exp)

#start plotting
fig_1 = plt.figure('fig_1')
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
plt.tick_params(top=True,bottom=True,left=True,right=False)
#start_line = 40

x_list_exp      = Ca_chain_r_exp[:,0]  # A of the O isotopes
y_list_exp      = Ca_chain_r_exp[:,1]  # experiment binding energy

x_list_DNNLO450 = Ca_chain_r_data[:,0]  # A of the O isotopes
y_list_DNNLO450 = Ca_chain_r_data[:,1]  # delta nnlo 450 calculation

x_list_NNLOsat  = Ca_chain_r_data[:,0]  # A of the O isotopes
y_list_NNLOsat  = Ca_chain_r_data[:,3]  # nnlosat calculation

print(x_list_NNLOsat)
print(y_list_NNLOsat)

x_list_magic    = Ca_chain_r_data[:,0]  # A of the O isotopes
y_list_magic    = Ca_chain_r_data[:,4]  # magic 1.8/2.0 calculation

x_list_DNNLO394 = Ca_chain_r_data[:,0]  # A of the O isotopes
y_list_DNNLO394 = Ca_chain_r_data[:,5]  # delta nnlo 394 calculation


l_exp         = plt.plot(x_list_exp,y_list_exp,color = 'k',linestyle = ':',linewidth=0.7, marker = 'x',markersize=5,label='Expt.',zorder=5) 
l_DNNLO450    = plt.scatter(x_list_DNNLO450,y_list_DNNLO450,color = 'g',s=30,marker = 's',label=r'$\Delta$NNLO$_{\rm{GO}}$(450)',zorder=4)
l_DNNLO394    = plt.scatter(x_list_DNNLO394,y_list_DNNLO394,color = 'b',s=30,marker = 'D',label=r'$\Delta$NNLO$_{\rm{GO}}$(394)',zorder=3)
l_NNLOsat     = plt.scatter(x_list_NNLOsat,y_list_NNLOsat,color = 'orange',s=30,marker = '^',label=r'NNLO$_{\rm{sat}}$',zorder=2)
l_magic       = plt.scatter(x_list_magic,y_list_magic,color = 'r',s=30,marker = 'p',label='1.8/2.0(EM)',zorder=1)

plt.ylabel(r'$R_{\rm{ch}}$[fm]',fontsize=18)
plt.xlabel('$A$',fontsize=18)
plt.yticks(np.arange(3.25,3.7,0.1),fontsize = 13)
plt.xticks(np.arange(40,55,2),fontsize = 14)
#plt.legend(loc=2, bbox_to_anchor=(1.63,0.5),borderaxespad = 0.)
plt.legend(loc=2,bbox_to_anchor=(0,1))

#plt.title('Charge radii of calcium isotopes')
plot_path = 'Ca_chain_Rch.pdf'
plt.savefig(plot_path)
plt.show()
plt.close("all")


