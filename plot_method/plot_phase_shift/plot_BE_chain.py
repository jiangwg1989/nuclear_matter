import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
import re
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
                raw_data[loop2][0] = float(temp_1[2])
                raw_data[loop2][1] = float(temp_1[3])
                raw_data[loop2][2] = float(temp_1[4])
                loop2 = loop2 + 1
            loop1 = loop1 + 1
       # print (loop2)

def input_file_count(file_path):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                loop2 = loop2 + 1
            loop1 = loop1 + 1
       # print ('data_num='+str(loop2))
        return loop2


def plot_BE(file_path):
   count = len(open(file_path,'rU').readlines())
   with open(file_path,'r') as f_1:
       data =  f_1.readlines()
       loop2 = 0
       loop1 = 0
       wtf = re.match('#', 'abc',flags=0)
       while loop1 < count:
           if ( re.match('#', data[loop1],flags=0) == wtf):
               temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
               raw_data[loop2][0] = float(temp_1[2])
               raw_data[loop2][1] = float(temp_1[3])
               raw_data[loop2][2] = float(temp_1[4])
               loop2 = loop2 + 1
           loop1 = loop1 + 1




#    print ('scattering_data_num='+str(scattering_data_num))
    raw_data_1 = np.zeros((scattering_data_num,4),dtype = np.float)
    input_residual_data(file_path,raw_data_1)
    loss = np.power(((raw_data_1[:,1] - raw_data_1[:,0])/raw_data_1[:,2]),2)
    tol_loss = np.sum(loss)/ len(loss)
    print ('loss='+str(tol_loss))


    fig_1 = plt.figure('fig_1')
    plt.subplots_adjust(wspace =0.5, hspace =0)



    ## pp 3F3
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.plot()
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 40
    x_list     = raw_data_1[start_line:start_line+8,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+8,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+8,0]  # theo
    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func(x_list_new) 

    l_exp      = plt.scatter(x_list,y_list_1,color = 'k',s = point_size,marker ='.',label='Granada PWA')
    l_theo     = plt.plot   (x_list_new,y_list_2_new,color = 'b',linestyle = '-',label='$\Delta$NNLO(450)') 

    plt.ylabel('$\delta(^3F_3)$(deg)')
    plt.xlabel(r'$T_{\rm{Lab}}$(MeV)')
    plt.yticks(np.arange(-4,0.5,1),fontsize = y_fontsize)
    plt.xticks(np.arange(0,251,50),fontsize = x_fontsize)
    plt.legend(loc=2, bbox_to_anchor=(1.63,0.5),borderaxespad = 0.)

    plt.suptitle('Proton-proton scattering phase shifts.')
    plot_path = 'pp_phase_shift.eps'
    plt.savefig(plot_path)
    plt.show()
    plt.close("all")




##########################################################
##########################################################
### setting parameters
##########################################################
##########################################################
theo_line          = 5
exp_line           = 7
exp_error_line     = 9
energy_line        = 1
E_max              = 200
data_points        = 8
plot_interpol_count= 1000
point_size         = 50
x_fontsize         = 8
y_fontsize         = 8
kind               = 'cubic' 
input_file_path = "./residual_data.txt"
plot_phase_shift(input_file_path)
