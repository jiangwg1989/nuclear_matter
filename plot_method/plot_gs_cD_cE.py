import numpy as np
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
                raw_data[loop2][0] = float(temp_1[0])
                raw_data[loop2][1] = float(temp_1[1])
                raw_data[loop2][2] = float(temp_1[2])
                raw_data[loop2][3] = float(temp_1[3])
                raw_data[loop2][4] = float(temp_1[4])
                loop2 = loop2 + 1
            loop1 = loop1 + 1
        print loop2

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
        return loop2




particle_num = 28
neutron_num  = 14
cE_min       = -1
cE_max       = 1
cE_gap       = 0.25
cE_count     = int( (cE_max-cE_min)/cE_gap +1)
cD_min       = -3
cD_max       = -3
cD_gap       = 0.5
cD_count     = int( (cD_max-cD_min)/cD_gap +1)
density_min  = 0.15 
density_max  = 0.23
density_gap  = 0.02
density_count= int( (density_max-density_min)/density_gap +1)

file_path = "cD-3.00-3.00_cE-1.00-1.00_c2_-0.49.dat"
data_num = input_file_count(file_path)
raw_data = np.zeros((data_num,5),dtype = np.float)
input_file_2(file_path,raw_data)


print ("data_num ="+str(data_num) )
#raw_data_new_4 = raw_data[np.where(raw_data[:,2] == 50 )]
#raw_data_new_3 = raw_data[np.where(raw_data[:,2] == 40 )]
#raw_data_new_2 = raw_data[np.where(raw_data[:,2] == 30 )]
#raw_data_new_1 = raw_data[np.where(raw_data[:,2] == 20 )]


#raw_data_new_5 = raw_data[np.where(raw_data[:,1] == 200)]
#temp_1 = raw_data_new_5[:,0]

#gs_converge = np.min(temp_1)

#gs_converge = -30.72325

#raw_data_new = np.zeros((200,2),dtype = np.float) 

#for loop in range(0,200):
#    raw_data_new[loop,1] = loop+4
#    raw_data_new[loop,0] = np.min(raw_data[np.where(raw_data[:,1]==loop+4)])

interpol_count  = 1000
saturation_point = np.zeros((data_num/density_count,2))
kind = "quadratic"
#print raw_data
raw_data_new = raw_data
#raw_data_new = raw_data[np.where(raw_data[:,0]==-3)]

print raw_data_new
for loop1 in range(len(raw_data_new)/density_count):
    x = raw_data_new[loop1*density_count:loop1*density_count+density_count,2]
    y = raw_data_new[loop1*density_count:loop1*density_count+density_count,3]
    spldens = np.linspace(density_min,density_max,num=interpol_count)
    print ("x="+str(x))
    print ("y="+str(y))
    f       = interpolate.interp1d(x,y,kind=kind)
    y_new   = np.zeros(interpol_count)
    y_new   = f(spldens) 
    saturation_point[loop1,1] = np.min(y_new)
    temp = spldens[np.where(y_new == np.min(y_new))]
    saturation_point[loop1,0] = temp[0]


saturation_point_2 = np.zeros((1,2))
raw_data_new_2 = raw_data[np.where((raw_data[:,0]==0) & (raw_data[:,1]==0))]
for loop1 in range(len(raw_data_new_2)/density_count):
    x = raw_data_new_2[loop1*density_count:loop1*density_count+density_count,2]
    y = raw_data_new_2[loop1*density_count:loop1*density_count+density_count,3]
    spldens = np.linspace(density_min,density_max,num=interpol_count)
    print ("x="+str(x))
    print ("y="+str(y))
    f       = interpolate.interp1d(x,y,kind=kind)
    y_new   = np.zeros(interpol_count)
    y_new   = f(spldens) 
    saturation_point_2[loop1,1] = np.min(y_new)
    temp = spldens[np.where(y_new == np.min(y_new))]
    saturation_point_2[loop1,0] = temp[0]


x_list = saturation_point[:,0]
y_list = saturation_point[:,1]
x_list_2 = saturation_point_2[:,0]
y_list_2 = saturation_point_2[:,1]


fig1 = plt.figure('fig1')
l1 = plt.scatter(x_list,y_list,color = 'b',s =10,marker ='x')
l2 = plt.scatter(x_list_2,y_list_2,color = 'r',s =30,marker ='x')
plt.xlim((0.1,0.22))
plt.ylim((-25,-10))
plot_path = 'saturation_point_cD_%.2f_%.2f_cE_%.2f_%.2f.eps' % (cD_min,cD_max,cE_min,cE_max)
plt.savefig(plot_path)
plt.show()


#
#
#
#
#x_list = raw_data_new[:,1]
#y_list = np.log10(raw_data_new[:,0] - gs_converge)
#
#x_list_1 = raw_data_new_1 [:,1]
#y_list_1 = np.log10(raw_data_new_1[:,0] - gs_converge)
#
#x_list_2 = raw_data_new_2 [:,1]
#y_list_2 = np.log10(raw_data_new_2[:,0] - gs_converge)
#
#x_list_3 = raw_data_new_3 [:,1]
#y_list_3 = np.log10(raw_data_new_3[:,0] - gs_converge)
#
#x_list_4 = raw_data_new_4 [:,1]
#y_list_4 = np.log10(raw_data_new_4[:,0] - gs_converge)
#
#
#
#
#fig_1 = plt.figure('fig_1')
#l1 = plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=20')
#l2 = plt.scatter(x_list_2,y_list_2,color='r',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=30')
#l3 = plt.scatter(x_list_3,y_list_3,color='g',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=40')
#l4 = plt.scatter(x_list_4,y_list_4,color='b',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=50')
#
#plt.title("E(converge)="+str(gs_converge))
#
#plt.ylabel("lg(E(infinte)-E(converge))")
#plt.legend(loc = 'lower left')
##plt.ylim((1.2,2.8))
##plt.savefig('Li6_radius_NN_prediction.jpg')
#plot_path = 'different_hw.eps'
#plt.savefig(plot_path)
##fig_1.show()
#
#
#
#fig_2 = plt.figure('fig_2')
#l = plt.scatter(x_list,y_list,color='k',linestyle='--',s = 10, marker = 'x', label='E(infinite)')
#
#
#plt.title("E(converge)="+str(gs_converge))
#plt.ylabel("lg(E(infinte)-E(converge))")
#plt.legend(loc = 'lower left')
##plt.ylim((1.2,2.8))
##plt.savefig('Li6_radius_NN_prediction.jpg')
#plot_path = 'lowest_each_Nmax.eps'
#plt.savefig(plot_path)
##fig_2.show()
#
#
#
#
#
##input()
