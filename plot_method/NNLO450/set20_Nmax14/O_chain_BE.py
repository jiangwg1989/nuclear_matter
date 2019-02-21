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
                raw_data[loop2][0] = float(temp_1[0])   # exp
                raw_data[loop2][1] = float(temp_1[1])   # cal
                raw_data[loop2][2] = float(temp_1[2])
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 


file_path    = "O_chain.txt"
data_num     =  4
O_chain_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,O_chain_data)


#start plotting
fig_1 = plt.figure('fig_1')
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
plt.tick_params(top=True,bottom=True,left=True,right=False)
#start_line = 40

x_list_DNNLO450 = O_chain_data[:,0]  # A of the O isotopes
x_list_exp      = O_chain_data[0:3,0]  # A of the O isotopes
y_list_exp      = O_chain_data[0:3,1]  # experiment binding energy
y_list_DNNLO450 = O_chain_data[:,2]  # delta nnlo 450 calculation


l_exp           = plt.plot(x_list_exp,y_list_exp,color='k', linestyle = '-.',linewidth=2,marker='o', markersize=8,zorder=2,label='Expt.')
l_DNNLO450      = plt.plot(x_list_DNNLO450,y_list_DNNLO450,color='b', linestyle = '-',linewidth=2,marker='^', markersize=8,alpha=0.7,zorder = 1 ,label='DNNLO$_{450}$')

plt.ylabel(r'$E_{\rm{g.s.}}$(MeV)',fontsize=16)
plt.xlabel('$A$',fontsize=16)
plt.yticks(np.arange(-175,-120,10),fontsize = 12)
plt.xticks(np.arange(16,28.5,2),fontsize = 13)
plt.legend(loc=3)

plt.title('Ground-state energies of oxygen')
plot_path = 'O_chain.eps'
plt.savefig(plot_path)
plt.show()
plt.close("all")


print('O_exp='+str(O_chain_data[:,2]))
