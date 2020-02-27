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
                raw_data[loop2][1] = float(temp_1[1])   # expt
                raw_data[loop2][2] = float(temp_1[2])   # NNLO450
                raw_data[loop2][3] = float(temp_1[3])   # NNLO394
                raw_data[loop2][4] = float(temp_1[4])  # 1.8/2.0
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 

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
                raw_data[loop2][1] = float(temp_1[1])   # expt
                loop2 = loop2 + 1 
            loop1 = loop1 + 1 



file_path    = "Ca_chain.txt"
data_num     =  8
O_chain_data = np.zeros((data_num,5),dtype = np.float)
input_file_2(file_path,O_chain_data)

file_path    = "Ca_chain_BE_exp.txt"
data_num     =  3
O_chain_data_exp = np.zeros((data_num,2),dtype = np.float)
input_file_1(file_path,O_chain_data_exp)





#start plotting
fig_1 = plt.figure('fig_1')
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
plt.tick_params(top=True,bottom=True,left=True,right=True)
#start_line = 40

x_list_exp      = O_chain_data[0:5,0].astype(np.int32)  # A of the O isotopes
y_list_exp      = O_chain_data[0:5,1]  # experiment binding energy

x_list_DNNLO450 = O_chain_data[:,0].astype(np.int32)  # A of the O isotopes
y_list_DNNLO450 = O_chain_data[:,2]  # delta nnlo 450 calculation

x_list_DNNLO394 = O_chain_data[:,0].astype(np.int32)  # A of the O isotopes
y_list_DNNLO394 = O_chain_data[:,3]  # delta nnlo 450 calculation

x_list_magic    = O_chain_data[:,0].astype(np.int32)  # A of the O isotopes
y_list_magic    = O_chain_data[:,4]  # delta nnlo 450 calculation

x_list_exp_new  = O_chain_data_exp[:,0].astype(np.int32)  # A of the O isotopes
y_list_exp_new  = O_chain_data_exp[:,1]  # experiment binding energy






l_exp           = plt.plot(x_list_exp,y_list_exp,color='k', linestyle = '',linewidth=0.5,marker='x', markersize=5,alpha=1,zorder=4,label='Expt.')
l_DNNLO450      = plt.plot(x_list_DNNLO450,y_list_DNNLO450,color='g', linestyle = '--',linewidth=0.5,marker='s', markersize=5,alpha=1,zorder=3,label=r'$\Delta$NNLO$_{\rm{GO}} $(450)')
l_DNNLO394      = plt.plot(x_list_DNNLO394,y_list_DNNLO394,color='b', linestyle = '',linewidth=0.5,marker='D', markersize=5,alpha=1,zorder=2,label=r'$\Delta$NNLO$_{\rm{GO}} $(394)')
l_magic         = plt.plot(x_list_magic,y_list_magic,color='red', linestyle = '--',linewidth=0.5,marker='p', markersize=5,alpha=1,zorder=1,label='1.8/2.0(EM)')
l_exp_new       = plt.plot(x_list_exp_new,y_list_exp_new,color='k', linestyle = '',linewidth=0.5,marker='x', markersize=5,alpha=1,zorder=5)


plt.ylabel(r'$E_{\rm{g.s.}}$(MeV)',fontsize=16)
plt.xlabel('$A$',fontsize=16)
#plt.xlim(())
plt.ylim((-460,-335))
plt.yticks(np.arange(-460,-339,20),fontsize = 12)
plt.xticks(np.arange(40,61.5,2),fontsize = 13)
#plt.legend(loc=2, bbox_to_anchor=(1.63,0.5),borderaxespad = 0.)
plt.legend(loc=3)
#plt.title('Ground-state energies of calcium')


left, bottom, width,height = 0.55,0.5,0.38,0.35
ax1 = fig_1.add_axes([left,bottom,width,height])
ax1.plot(x_list_exp,y_list_exp,color='k', linestyle = '',linewidth=0.5,marker='x', markersize=5,alpha=1,zorder=4,label='Expt.')
ax1.plot(x_list_DNNLO450,y_list_DNNLO450,color='g', linestyle = '',linewidth=0.5,marker='s', markersize=5,alpha=1,zorder=3,label=r'$\Delta$NNLO$_{\rm{GO}} $(450)')
ax1.plot(x_list_DNNLO394,y_list_DNNLO394,color='b', linestyle = '',linewidth=0.5,marker='D', markersize=5,alpha=1,zorder=2,label=r'$\Delta$NNLO$_{\rm{GO}} $(394)')
ax1.plot(x_list_magic,y_list_magic,color='red', linestyle = '',linewidth=0.5,marker='p', markersize=5,alpha=1,zorder=1,label='1.8/2.0(EM)')
ax1.plot(x_list_exp_new,y_list_exp_new,color='k', linestyle = '',linewidth=0.5,marker='x', markersize=5,alpha=1,zorder=5)
ax1.set_xlim(51,56)
ax1.set_ylim(-450,-436)
ax1.set_xticks(np.arange(51,57.1,1))
ax1.set_yticks(np.arange(-452,-435.9,4))



fig_1.tight_layout()
plot_path = 'Ca_chain.pdf'
plt.savefig(plot_path)
#plt.show()
plt.close("all")


