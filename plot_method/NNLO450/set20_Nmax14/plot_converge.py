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

x_list_e3max16 = Ca_chain_data[:,0]  # A of the O isotopes
y_list_e3max16 = Ca_chain_data[:,1]  # delta nnlo 450 calculation

x_list_e3max14 = Ca_chain_data[:,0]  # A of the O isotopes
y_list_e3max14 = Ca_chain_data[:,2]  # delta nnlo 450 calculation

x_list_e3max12 = Ca_chain_data[:,0]  # A of the O isotopes
y_list_e3max12 = Ca_chain_data[:,3]  # experiment binding energy


l_e3max16      = plt.plot(x_list_e3max16,y_list_e3max16,color='k', linestyle = '--',linewidth=1,marker='D', markersize=5,label='e3max16',zorder=5,alpha=0.8)
l_e3max14      = plt.plot(x_list_e3max16,y_list_e3max14,color='darkgreen', linestyle = '-.',linewidth=1,marker='D', markersize=5,label='e3max14',zorder=4,alpha=0.8)
l_e3max12      = plt.plot(x_list_e3max16,y_list_e3max12,color='yellowgreen', linestyle = ':' ,linewidth=1,marker='D', markersize=5,label='e3max12',zorder=3,alpha=0.8)

plt.fill_between(x_list_e3max16,y_list_e3max16,y_list_e3max12,facecolor='g',zorder=1,alpha=0.2)
plt.fill_between(x_list_e3max16,y_list_e3max16,y_list_e3max14,facecolor='g',zorder=2,alpha=0.6)


plt.ylabel(r'$E_{\rm{g.s.}}$(MeV)',fontsize=16)
plt.xlabel('$A$',fontsize=16)
plt.yticks(np.arange(-460,-319,20),fontsize = 12)
plt.xticks(np.arange(40,61.5,2),fontsize = 13)
plt.legend(loc=3)
#plt.title(r'Ground-state energies convergence of calcium with $\Delta$NNLO$_{\rm{GO}}$    (450) and $\Delta$NNLO$_{\rm{GO}}$(394)')
fig_1.tight_layout()

plot_path = 'convergence.pdf'
plt.savefig(plot_path)
#plt.show()
plt.close("all")


print('O_exp='+str(Ca_chain_data[:,2]))
