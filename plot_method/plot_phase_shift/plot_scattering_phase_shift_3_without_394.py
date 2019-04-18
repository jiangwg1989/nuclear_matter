import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
import re
from math import log
from math import e


def input_residual_data(file_path,raw_data):
    with open(file_path,'r') as f_2:
        count = len(open(file_path,'rU').readlines())
        data = f_2.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_2 = data[loop1][6:]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",temp_2)
                if ( float(temp_1[energy_line])*pow(10,int(temp_1[energy_line+1]))  <= E_max):
                    raw_data[loop2][0] = float(temp_1[theo_line])     * pow(10,int(temp_1[theo_line+1]))
                    raw_data[loop2][1] = float(temp_1[exp_line])      * pow(10,int(temp_1[exp_line+1] ))
                    raw_data[loop2][2] = float(temp_1[exp_error_line])* pow(10,int(temp_1[exp_error_line+1]))
                    raw_data[loop2][3] = float(temp_1[energy_line])   * pow(10,int(temp_1[energy_line+1]))
                #print raw_data[loop2][0] 
                    loop2 = loop2 + 1
            loop1 = loop1 + 1
#        print ('vec_input='+str(vec_input))
        print ('raw_data='+str(raw_data))

def input_NNLOsat_phase_data(file_path,raw_data,pw):
    with open(file_path,'r') as f_2:
        count = len(open(file_path,'rU').readlines())
        data = f_2.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.search(pw, data[loop1],flags=0) != wtf):
                while loop2< len(raw_data):
                    temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+8+loop2])
                    raw_data[loop2][0] = temp_1[0]
                    raw_data[loop2][1] = temp_1[2]
                    loop2 = loop2 + 1
            loop1 = loop1 + 1
#        print ('vec_input='+str(vec_input))
#        print ('raw_data='+str(raw_data))





def plot_phase_shift(file_path1,file_path2):
    with open(file_path1,'r') as f:
        count = len(open(file_path1,'rU').readlines())
        data = f.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count :
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_2 = data[loop1][6:]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",temp_2)
                if ( float(temp_1[energy_line])*pow(10,int(temp_1[energy_line+1]))  <= E_max):
                    loop2 = loop2 + 1
            loop1 = loop1 + 1
    scattering_data_num = loop2
    print ('scattering_data_num='+str(scattering_data_num))
    raw_data_1 = np.zeros((scattering_data_num,4),dtype = np.float)
    raw_data_2 = np.zeros((scattering_data_num,4),dtype = np.float)
    input_residual_data(file_path1,raw_data_1)
    input_residual_data(file_path2,raw_data_2)

    loss = np.power(((raw_data_1[:,1] - raw_data_1[:,0])/raw_data_1[:,2]),2)
    tol_loss = np.sum(loss)/ len(loss)
    print ('loss='+str(tol_loss))

    file_path = 'NNLOsat_phase.txt'
    
    NNLOsat_data = np.zeros((8,2))



    fig_2 = plt.figure('fig_2')
    plt.subplots_adjust(wspace =0.4, hspace =0)

    ## np 3D1
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(224)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 30*data_points
    x_list     = raw_data_1[start_line:start_line+ data_points  ,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+data_points ,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+data_points ,0]  # theo 450
    y_list_3   = raw_data_2[start_line:start_line+data_points ,0]  # theo 394

    input_NNLOsat_phase_data(file_path,NNLOsat_data,'3D1')
    y_list_4    = NNLOsat_data[:,1]


    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func_1       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func_1(x_list_new) 
    func_2       = interpolate.interp1d(x_list,y_list_3,kind=kind)
    y_list_3_new = func_2(x_list_new) 
    func_3       = interpolate.interp1d(x_list,y_list_4,kind=kind)
    y_list_4_new = func_3(x_list_new) 
 

    #print('y_list='+str(y_list_2))
    #print('y_list_2_new='+str(y_list_2_new))
    l_exp      = plt.plot(x_list,y_list_1,color = 'k',linestyle='',markersize=4, marker ='s',zorder=4,label='Granada PWA')
    l_theo_450 = plt.plot(x_list_new,y_list_2_new,color = 'g',linewidth=lw,linestyle = '--',zorder=3,label=r'$\Delta$NNLO$_{\rm{GO}}$(450)') 
    l_theo_394 = plt.plot(x_list_new,y_list_3_new,color = 'b',linewidth=lw,linestyle = '-.',zorder=2.5,label=r'$\Delta$NNLO$_{\rm{GO}}$(394)') 
#    l_theo_394 = plt.plot   (x_list_new,y_list_3_new,color = 'b',linestyle = '-.',zorder=2,label=r'$\Delta$NNLO$_{\rm{GO}}$(394)') 
    l_nnlosat  = plt.plot   (x_list_new,y_list_4_new,color = 'r',linewidth=lw,linestyle = ':',zorder=2,label=r'NNLO$_{\rm{sat}}$') 
    #plt.xlabel(fontsize = x_fontsize)
    plt.ylabel('$\delta(^3D_1)$(deg)',fontsize=ylabel_f)
    plt.xlabel(r'$T_{\rm{Lab}}$(MeV)',fontsize=xlabel_f)
    plt.yticks(np.arange(-30,11,10),fontsize = y_fontsize)
    plt.xticks(np.arange(0,201,50),fontsize = y_fontsize)
    plt.ylim((-30,9))
    #plt.ylim()

    #plt.legend(loc=2, bbox_to_anchor=(1.63,0.5),borderaxespad = 0.)


    ## np 3S1
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(223)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 28*data_points
    x_list     = raw_data_1[start_line:start_line+data_points,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+data_points,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+data_points,0]  # theo
    y_list_3   = raw_data_2[start_line:start_line+data_points ,0]  # theo 394

    input_NNLOsat_phase_data(file_path,NNLOsat_data,'3S1')
    y_list_4    = NNLOsat_data[:,1]


    for loop1 in range(len(y_list_1)):
        if y_list_1[loop1] < 0:
            y_list_1[loop1] = y_list_1[loop1]+180
    for loop2 in range(len(y_list_2)):
        if y_list_2[loop2] < 0:
            y_list_2[loop2] = y_list_2[loop2]+180
    for loop3 in range(len(y_list_3)):
        if y_list_3[loop3] < 0:
            y_list_3[loop3] = y_list_3[loop3]+180
    for loop4 in range(len(y_list_4)):
        if y_list_4[loop4] < 0:
            y_list_4[loop4] = y_list_4[loop4]+180
 
    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func_1       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func_1(x_list_new) 
    func_2       = interpolate.interp1d(x_list,y_list_3,kind=kind)
    y_list_3_new = func_2(x_list_new) 
    func_3       = interpolate.interp1d(x_list,y_list_4,kind=kind)
    y_list_4_new = func_3(x_list_new) 
    

    l_exp      = plt.plot(x_list,y_list_1,color = 'k',linestyle='',markersize=4, marker ='s',zorder=4,label='Granada PWA')
    l_theo_450 = plt.plot(x_list_new,y_list_2_new,color = 'g',linewidth=lw,linestyle = '--',zorder=3,label=r'$\Delta$NNLO$_{\rm{GO}}$(450)') 
    l_theo_394 = plt.plot(x_list_new,y_list_3_new,color = 'b',linewidth=lw,linestyle = '-.',zorder=2.5,label=r'$\Delta$NNLO$_{\rm{GO}}$(394)') 
#    l_theo_394 = plt.plot   (x_list_new,y_list_3_new,color = 'b',linestyle = '-.',zorder=2,label=r'$\Delta$NNLO$_{\rm{GO}}$(394)') 
    l_nnlosat = plt.plot(x_list_new,y_list_4_new,color = 'r',linewidth=lw,linestyle = ':',zorder=2,label=r'NNLO$_{\rm{sat}}$') 


    plt.ylabel('$\delta(^3S_1)$(deg)',fontsize=ylabel_f)
    plt.xlabel(r'$T_{\rm{Lab}}$(MeV)',fontsize=xlabel_f)
    plt.yticks(np.arange(-50,201,50),fontsize = y_fontsize)
    plt.xticks(np.arange(0,201,50),fontsize = y_fontsize)
    plt.ylim((-50,199))
   



    ## np 1S0
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(221)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 12*data_points
    x_list     = raw_data_1[start_line:start_line+data_points ,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+data_points,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+data_points,0]  # theo
    y_list_3   = raw_data_2[start_line:start_line+data_points ,0]  # theo 394
    input_NNLOsat_phase_data(file_path,NNLOsat_data,'np1S0')
    y_list_4    = NNLOsat_data[:,1]


    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func_1       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func_1(x_list_new) 
    func_2       = interpolate.interp1d(x_list,y_list_3,kind=kind)
    y_list_3_new = func_2(x_list_new) 
    func_3       = interpolate.interp1d(x_list,y_list_4,kind=kind)
    y_list_4_new = func_3(x_list_new) 

 
    #print('y_list='+str(y_list_2))
    #print('y_list_2_new='+str(y_list_2_new))
    l_exp      = plt.plot(x_list,y_list_1,color = 'k',linestyle='',markersize=4, marker ='s',zorder=4,label='Granada PWA')
    l_theo_450 = plt.plot(x_list_new,y_list_2_new,color = 'g',linewidth=lw,linestyle = '--',zorder=3,label=r'$\Delta$NNLO$_{\rm{GO}}$(450)') 
    l_theo_394 = plt.plot(x_list_new,y_list_3_new,color = 'b',linewidth=lw,linestyle = '-.',zorder=2.5,label=r'$\Delta$NNLO$_{\rm{GO}}$(394)') 
#    l_theo_394 = plt.plot   (x_list_new,y_list_3_new,color = 'b',linestyle = '-.',zorder=2,label=r'$\Delta$NNLO$_{\rm{GO}}$(394)') 
    l_nnlosat  = plt.plot(x_list_new,y_list_4_new,color = 'r',linewidth=lw,linestyle = ':',zorder=2,label=r'NNLO$_{\rm{sat}}$') 


    
    plt.legend(loc=4,bbox_to_anchor=(1,-0.22) ,fontsize=9,fancybox=True,facecolor='w',framealpha=1)
    #plt.xlabel(fontsize = x_fontsize)
    plt.ylabel('$\delta(^1S_0)$(deg)',fontsize=ylabel_f)
    plt.yticks(np.arange(-40,81,20),fontsize = y_fontsize)
    plt.xticks(np.arange(0,201,50),[])

    ## np 3P0
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(222)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 15 * data_points
    x_list     = raw_data_1[start_line:start_line+data_points,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+data_points,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+data_points,0]  # theo
    y_list_3   = raw_data_2[start_line:start_line+data_points ,0]  # theo 394
    input_NNLOsat_phase_data(file_path,NNLOsat_data,'np3P0')
    y_list_4    = NNLOsat_data[:,1]

    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func_1       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func_1(x_list_new) 
    func_2       = interpolate.interp1d(x_list,y_list_3,kind=kind)
    y_list_3_new = func_2(x_list_new) 
    func_3       = interpolate.interp1d(x_list,y_list_4,kind=kind)
    y_list_4_new = func_3(x_list_new) 


    l_exp      = plt.plot(x_list,y_list_1,color = 'k',linestyle='',markersize=4, marker ='s',zorder=4,label='Granada PWA')
    l_theo_450 = plt.plot(x_list_new,y_list_2_new,color = 'g',linewidth=lw,linestyle = '--',zorder=3,label=r'$\Delta$NNLO$_{\rm{GO}}$(450)') 
    l_theo_394 = plt.plot(x_list_new,y_list_3_new,color = 'b',linewidth=lw,linestyle = '-.',zorder=2.5,label=r'$\Delta$NNLO$_{\rm{GO}}$(394)') 
    l_nnlosat  = plt.plot(x_list_new,y_list_4_new,color = 'r',linewidth=lw,linestyle = ':',zorder=2,label=r'NNLO$_{\rm{sat}}$') 


    plt.ylabel('$\delta(^3P_0)$(deg)',fontsize=ylabel_f)
    plt.yticks(np.arange(-40,21,10),fontsize = y_fontsize)
    plt.xticks(np.arange(0,201,50),[])
    plt.legend(loc=4,bbox_to_anchor=(1,-0.22) ,fontsize=9,fancybox=True,facecolor='w',framealpha=1)
   # plt.legend(loc=2, bbox_to_anchor=(1.63,0.5),borderaxespad = 0.)
#    plt.suptitle('Neutron-proton scattering phase shifts (1).')

#    fig_2.tight_layout()
    plot_path = 'np_phase_shift_new.pdf'
    plt.savefig(plot_path)
#    plt.show()
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
ylabel_f           = 14
xlabel_f           = 14
x_fontsize         = 10
y_fontsize         = 10
lw                 = 3   #linewidth
kind               = 'cubic' 
input_file_path1 = "./residual_data_450.txt"
input_file_path2 = "./residual_data_394.txt"
plot_phase_shift(input_file_path1,input_file_path2)
