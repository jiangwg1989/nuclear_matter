import keras
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras import optimizers
from keras.optimizers import RMSprop
from scipy import interpolate
from keras.models import load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import layers
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.core import Lambda



def input_file_1(file_path,raw_data,cd_line,ce_line,dens_line,energy_line):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1]) 
                raw_data[loop2][0] = float(temp_1[cd_line])
                raw_data[loop2][1] = float(temp_1[ce_line])
                raw_data[loop2][2] = float(temp_1[dens_line])
                raw_data[loop2][3] = float(temp_1[energy_line])
                loop2 = loop2 + 1
            loop1 = loop1 + 1
        print loop2  

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
                loop2 = loop2 + 1
            loop1 = loop1 + 1
    #    print loop2





def input_raw_data_count(file_path):
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



def NN_all(input_path,output_path,data_num,monitor,min_delta,patience,epochs,input_dim,output_dim,interpol_count):
    #max_nmax_fit = 14
    raw_data = np.zeros((data_num,4),dtype = np.float)

    
    input_file_1(input_path,raw_data,cd_line,ce_line,dens_line,energy_line)
   

 
    #
    # To get more data, we do interpolation for the data
    #
    # interpolation for second colum
    # kind can be 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation     of first, second or third order)  
#    kind = "quadratic"
    X = []
    Y = []
    for i in range(0,raw_data.shape[0],5):
        dens = raw_data[i:i+5,2]
        temp = raw_data[i:i+5,3]
        spl_ccdt = interpolate.UnivariateSpline(dens,temp,k=4)
        spldens = np.linspace(dens[0],dens[len(dens)-1],num=interpol_count)
        interp = spl_ccdt(spldens)

        for j in range(0,spldens.size):
            X.append([raw_data[i,0],raw_data[i,1],spldens[j]])
            Y.append(interp[j])

    npX = np.array(X)
    npY = np.array(Y)

    data_interpolation = np.append(npX,np.transpose([npY]),1)
    print "npY"+str(npY)
    print "data_interpolation="+str(data_interpolation)

#    nmax_count = (nmax_max-nmax_min)/2 + 1
#    x_new = np.zeros((nmax_count,interpol_count))
#    y_new = np.zeros((nmax_count,interpol_count))
#    interpol_count_tot = 0   
#   
#    # interpolation for each namx 
#    for loop1 in range(0,nmax_count):
#        raw_data_new = raw_data[np.where(raw_data[:,1]==(loop1*2+nmax_min))]
#        x = raw_data_new[:,2]
#        y = raw_data_new[:,0]
#        x_new[loop1,:] = np.linspace(np.min(x),np.max(x),interpol_count)
#        f=interpolate.interp1d(x,y,kind=kind)
#        y_new[loop1,:] = f(x_new[loop1,:])
#        x_new_count = len(x_new[loop1,:])
#        interpol_count_tot = interpol_count_tot + x_new_count
# 
#    # joint all initerpoled data    
#    data_interpolation= np.zeros((interpol_count_tot,3))
#    count = 0
#    for loop2 in range(0,nmax_count):
#        for loop3 in range(0,interpol_count):
#            data_interpolation[count,1] = loop2*2 + nmax_min
#            data_interpolation[count,2] = x_new[loop2,loop3]
#            data_interpolation[count,0] = y_new[loop2,loop3]
#            count = count +1

   
    
    
    #
    # shuffle the data
    #
#    np.random.shuffle(data_interpolation)
#    np.random.shuffle(raw_data)
    
    #print len(raw_data)
    #print raw_data
    
    #batch_size = data_num
    
    input_shape = (input_dim,)
    input_data = Input(shape = input_shape)
    
    #raw_data_new  = raw_data[np.where(raw_data[:,1]<11)]
#    raw_data_new = data_interpolation[np.where(data_interpolation[:,1]<=max_nmax_fit)]
    data_new = data_interpolation[0:72000,:]
    #print "raw_data_new="+str(raw_data_new)
       
    x_train = data_new[:,0:3]
    y_train = data_new[:,3]
    
    #print "x_train = "+str(x_train)
    #print "y_train = "+str(y_train)
    
    #
    # NN Model
    # 
    x = Dense(16, activation = 'sigmoid')(input_data)
    #x = Dense(8, activation = 'sigmoid')(x)
    #x = Dense(8, activation = 'sigmoid')(x)
    #x = Dense(8, activation = 'sigmoid')(x)
    predictions = Dense(output_dim)(x)
   

   # model = Model(inputs= input_data, outputs = predictions)
   # 
   # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
   # 
   # model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
   # 
   # early_stopping = EarlyStopping(monitor=monitor,min_delta = min_delta , patience=patience, verbose=0, mode='min')
   # 
   # history = model.fit(x_train,y_train, epochs = epochs, validation_split = 0.01 , shuffle = 1, callbacks=[early_stopping])
   # loss = history.history['loss'][len(history.history['loss'])-1]
   # val_loss = history.history['val_loss'][len(history.history['val_loss'])-1]
   # 
   # fig4 = plt.figure('fig4')
   # plt.plot(history.history['loss'])
   # plt.plot(history.history['val_loss'])
   # plt.savefig('loss_val_loss.eps')
   # plt.close('all')


   # model_path = 'test.h5' 
   # model.save(model_path)
   
    #
    # load model
    #
    model = load_model('test.h5') 
    
    
    
    count = len(range(4,204,1))*len(range(5,121,1))
    x_test = np.zeros((count,2),dtype = np.float)
    
    loop3 = 0
    
    
   #for loop1 in range(0,1,0.1):
   #    for loop2 in range(1,4,0.25):
   #        x_test[loop3][0] = loop1 
   #        x_test[loop3][1] = loop2 
   #        loop3 = loop3 + 1
    
    #print x_test
    x_test = data_interpolation[:,0:3] 
    y_test = model.predict(x_test)
    
    raw_predic_data = np.concatenate((x_test,y_test),axis=1)
    
    #print "raw_predic_data="+str(raw_predic_data)
    
    #fig,(ax0,ax1) = plt.subplots(nrows = 2, figsize=(9,9))
    
    x_list_1 = raw_data[:,2] 
    y_list_1 = raw_data[:,3]
    #print "x_list_1"+str(x_list_1)
    #print "y_list_1"+str(y_list_1)
    print "size()="+str(len(data_interpolation))
    
#    raw_predic_data_4 = raw_predic_data[np.where(raw_predic_data[:,1]==4)]
#    raw_predic_data_8 = raw_predic_data[np.where(raw_predic_data[:,1]==8)]
#    raw_predic_data_12 = raw_predic_data[np.where(raw_predic_data[:,1]==12)]
#    raw_predic_data_20 = raw_predic_data[np.where(raw_predic_data[:,1]==20)]
#    raw_predic_data_40 = raw_predic_data[np.where(raw_predic_data[:,1]==40)]
#    raw_predic_data_60 = raw_predic_data[np.where(raw_predic_data[:,1]==100)]
    
#    temp = (raw_predic_data[np.where(raw_predic_data[:,1]== 200)])
#    #gs_converge = np.min(temp[:,0])
#   
# 
#    
#    x_list_2 = raw_predic_data_4[:,2]
#    y_list_2 = raw_predic_data_4[:,0]
#    
#    x_list_3 = raw_predic_data_8[:,2]
#    y_list_3 = raw_predic_data_8[:,0]
#    
#    x_list_4 = raw_predic_data_12[:,2]
#    y_list_4 = raw_predic_data_12[:,0]
#    
#    x_list_5 = raw_predic_data_20[:,2]
#    y_list_5 = raw_predic_data_20[:,0]
#    
#    
#    x_list_6 = raw_predic_data_40[:,2]
#    y_list_6 = raw_predic_data_40[:,0]
#    
#    x_list_7 = raw_predic_data_60[:,2]
#    y_list_7 = raw_predic_data_60[:,0]
    
    
    fig1 = plt.figure('fig1')
   # ax = plt.subplot(111)
    l1=plt.scatter(x_list_1[0:5],y_list_1[0:5],color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    l1=plt.scatter(x_list_1[100:105],y_list_1[100:105],color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    l1=plt.scatter(x_list_1[200:205],y_list_1[200:205],color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    l1=plt.scatter(x_list_1[300:305],y_list_1[300:305],color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    l1=plt.scatter(x_list_1[360:365],y_list_1[360:365],color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    l1=plt.scatter(x_list_1[400:405],y_list_1[400:405],color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    l1=plt.scatter(x_list_1[440:445],y_list_1[440:445],color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    l1=plt.scatter(x_list_1[470:475],y_list_1[470:475],color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    l2=plt.plot(x_test[0:1000,2],y_test[0:1000],color='y',linestyle='--',label='NN_Nmax_4')
    l2=plt.plot(x_test[20000:21000,2],y_test[20000:21000],color='g',linestyle='--',label='NN_Nmax_4')
    l2=plt.plot(x_test[40000:41000,2],y_test[40000:41000],color='b',linestyle='--',label='NN_Nmax_4')
    l2=plt.plot(x_test[60000:61000,2],y_test[60000:61000],color='r',linestyle='--',label='NN_Nmax_4')
    l2=plt.plot(x_test[72000:73000,2],y_test[72000:73000],color='c',linestyle='--',label='NN_Nmax_4')
    l2=plt.plot(x_test[80000:81000,2],y_test[80000:81000],color='c',linestyle='--',label='NN_Nmax_4')
    l2=plt.plot(x_test[88000:89000,2],y_test[88000:89000],color='c',linestyle='--',label='NN_Nmax_4')
    l2=plt.plot(x_test[94000:95000,2],y_test[94000:95000],color='c',linestyle='--',label='NN_Nmax_4')


#    l3=plt.plot(x_list_3,y_list_3,color='r',linestyle='--',label='NN_Nmax_8')
#    l4=plt.plot(x_list_4,y_list_4,color='g',linestyle='--',label='NN_Nmax_12')
#    l5=plt.plot(x_list_5,y_list_5,color='c',linestyle='--',label='NN_Nmax_20')
    
#    l6=plt.plot(x_list_6,y_list_6,color='m',linestyle='--',label='NN_Nmax_40')
#    l7=plt.plot(x_list_7,y_list_7,color='b',linestyle='--',label='NN_Nmax_100')
    #l4=fig1.scatter(x_list_2,y_list_2,color='y',linestyle='--',marker=',')
    #l5=fig1.scatter(x_list_2,y_list_2,color='r',linestyle='--',marker=',')
    #l6=fig1.scatter(x_list_2,y_list_2,color='c',linestyle='--',marker=',')
    #fig1.scatter(x_list_2,y_list_2,color='m',linestyle='--',marker=',')
    #plt.legend(loc = 'upper left')

#    xmajorLocator   = MultipleLocator(10)
#    #xmajorFormatter = FormatStrFormatter('%5f')
#    xminorLocator   = MultipleLocator(2)
#    
#    
#    ymajorLocator   = MultipleLocator(1)
#    ymajorFormatter = FormatStrFormatter('%1.1f')
#    yminorLocator   = MultipleLocator(1)
#
#    ax.xaxis.set_major_locator(xmajorLocator)
#    #ax.xaxis.set_major_formatter(xmajorFormatter)
#    ax.yaxis.set_major_locator(ymajorLocator)
#    ax.yaxis.set_major_formatter(ymajorFormatter)
#    ax.xaxis.set_minor_locator(xminorLocator)
#    ax.yaxis.set_minor_locator(yminorLocator)
#    ax.xaxis.grid(True, which='major') 
#    ax.yaxis.grid(True, which='minor')
    

#   plt.legend(loc='upper right', bbox_to_anchor=(1.5,0.75),ncol=1,fancybox=True,shadow=True,borderaxespad = 0.)
    #plt.title("gs(infinite)="+str(gs_converge))
    plot_path = 'test.eps'
   # plt.ylim((-29,-23))  
   # plt.xlim((10,50))
#    plt.subplots_adjust(right = 0.7)
    plt.xlabel(r"$\rho$ (fm$^{-3}$)",fontsize=14)
    plt.ylabel('E/A (MeV)',fontsize=14)
    plt.savefig(plot_path)
#    plt.close('all')
    fig1.show()
    
    
#    file_path = "gs_NN_prediction.txt"
#    with open(file_path,'w') as f_1:
#        for loop1 in range(1,count):
#            f_1.write('{:>-10.5f}'.format(y_test[loop1,0]))
#            f_1.write('{:>10}'.format(x_test[loop1,0]))
#            f_1.write('{:>10}'.format(x_test[loop1,1])+'\n')
#    os.system('cp '+file_path+' '+output_path)        
    os.system('cp '+model_path+' '+output_path) 
    os.system('cp '+'gs.eps'+' '+output_path) 
#    os.system('cp '+'loss_val_loss.eps'+' '+output_path) 
##
## plot different_hw.eps and lowest_each_Nmax.eps
##
#    data_num = len(open(file_path,'rU').readlines())
#    raw_data = np.zeros((data_num,3),dtype = np.float)
#    input_file_2(file_path,raw_data)
#    
#    raw_data_new_4 = raw_data[np.where(raw_data[:,2] == 50 )]
#    raw_data_new_3 = raw_data[np.where(raw_data[:,2] == 40 )]
#    raw_data_new_2 = raw_data[np.where(raw_data[:,2] == 30 )]
#    raw_data_new_1 = raw_data[np.where(raw_data[:,2] == 20 )]
#    
#    raw_data_new_5 = raw_data[np.where(raw_data[:,1] == 200)]
#    temp_1 = raw_data_new_5[:,0]
#    #gs_converge = np.min(temp_1)
#    
#    raw_data_new = np.zeros((200,2),dtype = np.float)
#    for loop in range(0,200):
#        raw_data_new[loop,1] = loop+4
#        raw_data_new[loop,0] = np.min(raw_data[np.where(raw_data[:,1]==loop+4)])
#    
#    x_list = raw_data_new[:,1]
#    y_list = np.log10(raw_data_new[:,0] - gs_converge)
#    
#    x_list_1 = raw_data_new_1 [:,1]
#    y_list_1 = np.log10(raw_data_new_1[:,0] - gs_converge)
#    
#    x_list_2 = raw_data_new_2 [:,1]
#    y_list_2 = np.log10(raw_data_new_2[:,0] - gs_converge)
#    
#    x_list_3 = raw_data_new_3 [:,1]
#    y_list_3 = np.log10(raw_data_new_3[:,0] - gs_converge)
#    
#    x_list_4 = raw_data_new_4 [:,1]
#    y_list_4 = np.log10(raw_data_new_4[:,0] - gs_converge)
#    fig_1 = plt.figure('fig_1')
#    l1 = plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=20')
#    l2 = plt.scatter(x_list_2,y_list_2,color='r',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=30')
#    l3 = plt.scatter(x_list_3,y_list_3,color='g',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=40')
#    l4 = plt.scatter(x_list_4,y_list_4,color='b',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=50')
#    
#    plt.title("E(converge)="+str(gs_converge))
#    
#    plt.ylabel("lg(E(infinte)-E(converge))")
#    plt.legend(loc = 'lower left')
#    #plt.ylim((1.2,2.8))
#    #plt.savefig('Li6_radius_NN_prediction.jpg')
#    plot_path = 'different_hw.eps'
#    plt.savefig(plot_path)
#    #fig_1.show()
#    fig_2 = plt.figure('fig_2')
#    l = plt.scatter(x_list,y_list,color='k',linestyle='--',s = 10, marker = 'x', label='E(infinite)')
#    
#    
#    plt.title("E(converge)="+str(gs_converge))
#    plt.ylabel("lg(E(infinte)-E(converge))")
#    plt.legend(loc = 'lower left')
#    #plt.ylim((1.2,2.8))
#    #plt.savefig('Li6_radius_NN_prediction.jpg')
#    plot_path = 'lowest_each_Nmax.eps'
#    plt.savefig(plot_path)
#    #fig_2.show()
#    plt.close('all') 
#   # import plot_gs as plot
#    os.system('cp '+'different_hw.eps'+' '+output_path) 
#    os.system('cp '+'lowest_each_Nmax.eps'+' '+output_path) 
    #return gs_converge,loss,val_loss



#
# all NN parameters
#
nuclei = 'He4'
target_option = 'gs'
input_path = 'cE0.2-1cd2.0-4.0.dat'
#output_path = './result/gs/'
data_num = input_raw_data_count(input_path)
print 'data_num='+str(data_num)
# earlystopping parameters
monitor  = 'loss'
min_delta = 0.00001
patience = 10
epochs = 10000
input_dim = 3 
output_dim = 1
# interpolation setting
interpol_count = 1000
cd_line = 0
ce_line = 1
dens_line = 3
energy_line = 7
run_times_start = 1 
run_times_end   = 1


gs_converge_all = np.zeros(run_times_end)
loss_all = np.zeros(run_times_end)
val_loss_all = np.zeros(run_times_end)

output_path = "./result"
NN_all(input_path=input_path,output_path=output_path,data_num=data_num,monitor=monitor,min_delta=min_delta,patience=patience,epochs=epochs,input_dim=input_dim,output_dim=output_dim,interpol_count=interpol_count)


#os.system('mkdir '+nuclei)
#os.system('mkdir '+nuclei+'/'+target_option)        


#for max_nmax_fit in range(18,19,2):
#    os.system('mkdir '+nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit))
#    with open(nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+'gs_NN_info.txt','a') as f_3:
#        #f_3.read()
#        f_3.write('#############################################'+'\n')
#        f_3.write('# loop   gs_energy       loss       val_loss'+'\n')
#   for loop_all in range(run_times_start-1,run_times_end):
#        os.system('mkdir '+nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1))
#        output_path = nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1)
#        gs_converge_all[loop_all],loss_all[loop_all],val_loss_all[loop_all] = NN_all(input_path=input_path,output_path=output_path,data_num=data_num,monitor=monitor,min_delta=min_delta,patience=patience,epochs=epochs,input_dim=input_dim,output_dim=output_dim,interpol_count=interpol_count,max_nmax_fit=max_nmax_fit)
#        with open(nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+'gs_NN_info.txt','a') as f_3:
#            #f_3.read()
#            f_3.write('{:>5}'.format(loop_all+1)+'   ')
#            f_3.write('{:>-10.5f}'.format(gs_converge_all[loop_all])+'   ')
#            f_3.write('{:>-10.5f}'.format(loss_all[loop_all])+'   ')
#            f_3.write('{:>-10.5f}'.format(val_loss_all[loop_all])+'\n')
#
#
#    
#print 'gs_converge_all='+str(gs_converge_all)


input()
