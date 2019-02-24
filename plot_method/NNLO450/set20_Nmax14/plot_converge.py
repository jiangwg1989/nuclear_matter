import numpy as np
import matplotlib
#matplotlib.use('PS')
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
                raw_data[loop2][1] = float(temp_1[1])   # e3max16
                raw_data[loop2][2] = float(temp_1[2])   # e3max14
                raw_data[loop2][3] = float(temp_1[3])   # e3max12
                loop2 = loop2 + 1                        
            loop1 = loop1 + 1 


file_path    = "converge_450.txt"
data_num     =  5
Ca_chain_data = np.zeros((data_num,4),dtype = np.float)
input_file_2(file_path,Ca_chain_data)


#start plotting
fig_1 = plt.figure('fig_1')
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
plt.tick_params(top=True,bottom=True,left=True,right=False)
#start_line = 40

x_list_DNNLO450 = Ca_chain_data[:,0]  # A of the O isotopes
y_list_DNNLO450 = Ca_chain_data[:,2]  # delta nnlo 450 calculation

x_list_DNNLO394 = Ca_chain_data[:,0]  # A of the O isotopes
y_list_DNNLO394 = Ca_chain_data[:,3]  # delta nnlo 450 calculation

x_list_exp      = Ca_chain_data[0:3,0]  # A of the O isotopes
y_list_exp      = Ca_chain_data[0:3,1]  # experiment binding energy


l_exp           = plt.plot(x_list_exp,y_list_exp,color='k', linestyle = '-.',linewidth=1,marker='o', markersize=5,zorder=3,label='Expt.')
l_DNNLO450      = plt.plot(x_list_DNNLO450,y_list_DNNLO450,color='g', linestyle = '--',linewidth=1,marker='D', markersize=5,alpha=0.7,zorder = 1 ,label=r'$\Delta$NNLO$_{\rm{GO}}$(450)')
l_DNNLO394      = plt.plot(x_list_DNNLO394,y_list_DNNLO394,color='b', linestyle = ':',linewidth=1,marker='D', markersize=5,alpha=0.7,zorder = 2 ,label=r'$\Delta$NNLO$_{\rm{GO}}$(394)')

plt.ylabel(r'$E_{\rm{g.s.}}$(MeV)',fontsize=16)
plt.xlabel('$A$',fontsize=16)
plt.yticks(np.arange(-460,-339,20),fontsize = 12)
plt.xticks(np.arange(40,61.5,2),fontsize = 13)
plt.legend(loc=3)
plt.title(r'Ground-state energies convergence of calcium with $\Delta$NNLO$_{\rm{GO}}$    (450) and $\Delta$NNLO$_{\rm{GO}}$(394)')
fig_1.tight_layout()

plot_path = 'convergence.eps'
plt.savefig(plot_path)
#plt.show()
plt.close("all")


print('O_exp='+str(Ca_chain_data[:,2]))
