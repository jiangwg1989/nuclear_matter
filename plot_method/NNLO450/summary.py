#coding=utf-8  
import re
import numpy as np
import math
import xlwt  
import xlrd  

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



def read_file(input_path):
    LECs = np.zeros(17)
    few_body = np.zeros(10)
    BE  = np.zeros(4)
    BE_flag = 0 
    radii = np.zeros(2)
    radii_flag = 0
    two_plus = np.zeros(2)
    two_plus_flag = 0
    with open(input_path+'info.txt','r') as f_1:
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop in range(len(data)):
            if ( re.search('-0.74', data[loop],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop])
                LECs[0:6] = temp_1[0:6]
                temp_2 = re.findall(r"[-+]?\d+\.?\d*",data[loop+1])
                LECs[6:12] = temp_2[0:6]
                temp_3 = re.findall(r"[-+]?\d+\.?\d*",data[loop+2])
                LECs[12:17] = temp_3[0:5]
            if ( re.search('nucleon_scattering', data[loop],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop])
                scattering_loss=temp_1[0]
            if ( re.search('pnm =', data[loop],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop])
                pnm_14 = temp_1[0]  
                dens   = temp_1[1]                
                snm_28 = temp_1[2]
            if ( re.search('H2 BINDING ENERGY', data[loop],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop])
                few_body[0] = temp_1[1]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop+4])
                few_body[1] = temp_1[1]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop+8])
                few_body[2] = temp_1[1]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop+12])
                few_body[3] = temp_1[1]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop+13])
                few_body[4] = temp_1[1]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop+17])
                few_body[5] = temp_1[1]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop+21])
                few_body[6] = temp_1[1]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop+25])
                few_body[7] = temp_1[1]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop+29])
                few_body[8] = temp_1[1]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop+33])
                few_body[9] = temp_1[1]
            if ( (re.search('CCSD', data[loop],flags=0) != wtf) & (BE_flag==0)):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop])
                BE[0] = temp_1[1]
                temp_2 = re.findall(r"[-+]?\d+\.?\d*",data[loop+1])
                BE[1] = temp_2[1]
                temp_3 = re.findall(r"[-+]?\d+\.?\d*",data[loop+2])
                BE[2] = temp_3[1]
                temp_4 = re.findall(r"[-+]?\d+\.?\d*",data[loop+3])
                BE[3] = temp_4[1]
                BE_flag = 1
            if ( (re.search('Full expectation value', data[loop],flags=0) != wtf) & (radii_flag==0)):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop])
                radii[0] = temp_1[3]
                temp_2 = re.findall(r"[-+]?\d+\.?\d*",data[loop+1])
                radii[1] = temp_2[3]
                radii_flag = 1
#            if ( (re.search('2 +', data[loop],flags=0) != wtf) & (two_plus_flag==0)):
#                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop])
#                two_plus[0] = temp_1[2]
#                print(data[])
#                temp_2 = re.findall(r"[-+]?\d+\.?\d*",data[loop+1])
#                two_plus[1] = temp_2[2]
#                two_plus_flag = 1

    file_path  = input_path+"N_132.txt"
    data_num   = input_file_count(file_path)
    N_132_data = np.zeros((data_num,3),dtype = np.float)
    N_28_data  = np.zeros((data_num,3),dtype = np.float)
    input_file_2(file_path,N_132_data)
    
    file_path  = input_path+"N_28.txt"
    input_file_2(file_path,N_28_data)
    interpol_count = 1000

    X  = []
    Y1 = []
    Y2 = []
    for i in range(0,N_132_data.shape[0],5):
        dens = N_132_data[i:i+5,0]
        temp_snm = N_132_data[i:i+5,1]
        temp_pnm = N_132_data[i:i+5,2]
        spl_ccdt_snm = interpolate.UnivariateSpline(dens,temp_snm,k=4)
        spl_ccdt_pnm = interpolate.UnivariateSpline(dens,temp_pnm,k=4)
        spldens = np.linspace(dens[0],dens[len(dens)-1],num=interpol_count)
        interp_snm = spl_ccdt_snm(spldens)
        interp_pnm = spl_ccdt_pnm(spldens)
        for j in range(0,spldens.size):
                X.append(spldens[j])
                Y1.append(interp_snm[j])
                Y2.append(interp_pnm[j])
    npX  = np.array(X)
    npY1 = np.array(Y1)
    npY2 = np.array(Y2)
    
    
    N_132_interpolation = np.append(np.transpose([npX]),np.transpose([npY1]),1)
    N_132_interpolation = np.append(N_132_interpolation,np.transpose([npY2]),1)
    #print ("npY1"+str(npY1))
    print ("data_interpolation="+str(N_132_interpolation))
    #data_interpolation_backup = data_interpolation.copy()
    
    
    X  = []
    Y1 = []
    Y2 = []
    for i in range(0,N_28_data.shape[0],5):
        dens = N_28_data[i:i+5,0]
        temp_snm = N_28_data[i:i+5,1]
        temp_pnm = N_28_data[i:i+5,2]
        spl_ccdt_snm = interpolate.UnivariateSpline(dens,temp_snm,k=4)
        spl_ccdt_pnm = interpolate.UnivariateSpline(dens,temp_pnm,k=4)
        spldens = np.linspace(dens[0],dens[len(dens)-1],num=interpol_count)
        interp_snm = spl_ccdt_snm(spldens)
        interp_pnm = spl_ccdt_pnm(spldens)
        for j in range(0,spldens.size):
                X.append(spldens[j])
                Y1.append(interp_snm[j])
                Y2.append(interp_pnm[j])
    npX  = np.array(X)
    npY1 = np.array(Y1)
    npY2 = np.array(Y2)
    
    
    N_28_interpolation = np.append(np.transpose([npX]),np.transpose([npY1]),1)
    N_28_interpolation = np.append(N_28_interpolation,np.transpose([npY2]),1)
    
    
    #data analysis
    N_28_saturation_snm = np.min(N_28_interpolation[:,1])
    temp1 = N_28_interpolation[np.where(N_28_interpolation[:,1]==N_28_saturation_snm),0]
    N_28_saturation_dens = temp1[0]
    temp2 = N_28_interpolation[np.where(N_28_interpolation[:,1]==N_28_saturation_snm),2]
    N_28_saturation_pnm = temp2[0]
    saturation_energy_28 = N_28_saturation_pnm - N_28_saturation_snm
    
    #df_28 = np.diff(N_28_interpolation[:,1])/np.diff(N_28_interpolation[:,0])
    #ddf_28 = np.diff(df_28) /np.diff(N_28_interpolation[1:len(N_28_interpolation),0])
    #temp3 = ddf_28[np.where(N_28_interpolation[:,1]==N_28_saturation_snm)]
    #ddf_saturation_dens_28 = temp3[0]
    #print ('ddf_saturation_dens_28=',ddf_saturation_dens_28)
    #print ('saturation_dens_28=',N_28_saturation_dens)
    #K0 = 9* pow(N_28_saturation_dens,2)*ddf_saturation_dens_28
    #print ('K0_28=',K0)
    
    N_132_saturation_snm = np.min(N_132_interpolation[:,1])
    temp1 = N_132_interpolation[np.where(N_132_interpolation[:,1]==N_132_saturation_snm),0]
    N_132_saturation_dens = temp1[0]
    temp2 = N_132_interpolation[np.where(N_132_interpolation[:,1]==N_132_saturation_snm),2]
    N_132_saturation_pnm = temp2[0]
#    print ('snm='+str(N_132_saturation_snm))
#    print ('dens='+str(N_132_saturation_dens))
#    print ('pnm='+str(N_132_saturation_pnm))
    symetric_energy_132 = N_132_saturation_pnm- N_132_saturation_snm
#    print ('saturation_energy='+str(saturation_energy_132))
    

    return LECs,few_body,BE,radii,two_plus,N_132_saturation_snm,N_132_saturation_dens,N_132_saturation_pnm,symetric_energy_132


#    with open(output_path,'a') as f_2:
#        for loop1 in range(len(LECs)):
#            f_2.write(str(LECs[loop1])+'  ')
#        for loop1 in range(len(few_body)):
#            f_2.write(str(few_body[loop1])+'  ')
#        for loop1 in range(len(BE)):
#            f_2.write(str(BE[loop1])+'  ')
#        for loop1 in range(len(radii)):
#            f_2.write(str(radii[loop1])+'  ')
#        f_2.write('\n')

def write_excel(filename,sheet,num,input_path):
    temp_1 = re.findall(r"[-+]?\d+\.?\d*",input_path)
    name = temp_1[0]
    sheet.write(num+1,0,name)
    pointer = 1
    for loop1 in range(len(LECs)):
        sheet.write(num+1,loop1+1,LECs[loop1])
        pointer = pointer + 1
    for loop1 in range(len(few_body)):
        sheet.write(num+1,loop1+pointer+1,few_body[loop1])
    pointer = pointer + len(few_body)
    for loop1 in range(len(BE)):
        sheet.write(num+1,loop1+pointer+2,BE[loop1])
    pointer = pointer + len(BE)
    for loop1 in range(len(radii)):
        sheet.write(num+1,loop1+pointer+3,np.sqrt(radii[loop1]))
    pointer = pointer + len(radii)
    sheet.write(num+1,loop1+pointer+3,snm)
    sheet.write(num+1,loop1+pointer+4,dens[0])
    sheet.write(num+1,loop1+pointer+5,pnm[0])
    sheet.write(num+1,loop1+pointer+6,S_energy[0])
#    f_2.write('\n')





output_path= './summary.xls'

LECs = np.zeros(17)
few_body = np.zeros(10)
BE  = np.zeros(4)
radii = np.zeros(2)
two_plus = np.zeros(2)

filename=xlwt.Workbook()  
sheet1=filename.add_sheet("NNLO450")  



sheet1.write(1,0,'set_num')
sheet1.write(1,1,'c1')
sheet1.write(1,2,'c2')
sheet1.write(1,3,'c3')
sheet1.write(1,4,'c4')

sheet1.write(1,5,'Ct_1s0np')
sheet1.write(1,6,'Ct_1S0nn')
sheet1.write(1,7,'Ct_1s0pp')
sheet1.write(1,8,'Ct_3S1(pp,nn,np)')
sheet1.write(1,9,'C_1S0')
sheet1.write(1,10,'C_3P0')
sheet1.write(1,11,'C_3P1')
sheet1.write(1,12,'C_3P2')
sheet1.write(1,13,'C_1p1')
sheet1.write(1,14,'C_3S1')
sheet1.write(1,15,'C3S1-3D1')
sheet1.write(1,16,'cD')
sheet1.write(1,17,'cE')

sheet1.write(1,19,'E(H2)')
sheet1.write(1,20,'R_p(H2)')
sheet1.write(1,21,'Q(H2)')
sheet1.write(1,22,'P_D-state(H2)')
sheet1.write(1,23,'E(H3)')
sheet1.write(1,24,'R_p(H3)')
sheet1.write(1,25,'E(He3)')
sheet1.write(1,26,'R_p(He3)')
sheet1.write(1,27,'E(He4)')
sheet1.write(1,28,'R_p(He4)')


sheet1.write(1,30,'Lambda-CCSD(T)(O16)')
sheet1.write(1,31,'Lambda-CCSD(T)(O24)')
sheet1.write(1,32,'Lambda-CCSD(T)(Ca40)')
sheet1.write(1,33,'Lambda-CCSD(T)(Ca48)')

sheet1.write(1,35,'R_p(O16)')
sheet1.write(1,36,'R_p(Ca40)')

sheet1.write(1,38,'snm_E/A')
sheet1.write(1,39,'saturation_density')
sheet1.write(1,40,'pnm_E/A')
sheet1.write(1,41,'symmetric_energy')
num = 1
input_path = './set16_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set17_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set18_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set19_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set20_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set21_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set22_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1


sheet1.write(num+2+1,0,'set_num')
sheet1.write(num+2+1,1,'c1')
sheet1.write(num+2+1,2,'c2')
sheet1.write(num+2+1,3,'c3')
sheet1.write(num+2+1,4,'c4')

sheet1.write(num+2+1,5,'Ct_1s0np')
sheet1.write(num+2+1,6,'Ct_1S0nn')
sheet1.write(num+2+1,7,'Ct_1s0pp')
sheet1.write(num+2+1,8,'Ct_3S1(pp,nn,np)')
sheet1.write(num+2+1,9,'C_1S0')
sheet1.write(num+2+1,10,'C_3P0')
sheet1.write(num+2+1,11,'C_3P1')
sheet1.write(num+2+1,12,'C_3P2')
sheet1.write(num+2+1,13,'C_1p1')
sheet1.write(num+2+1,14,'C_3S1')
sheet1.write(num+2+1,15,'C3S1-3D1')
sheet1.write(num+2+1,16,'cD')
sheet1.write(num+2+1,17,'cE')

sheet1.write(num+2+1,19,'E(H2)')
sheet1.write(num+2+1,20,'R_p(H2)')
sheet1.write(num+2+1,21,'Q(H2)')
sheet1.write(num+2+1,22,'P_D-state(H2)')
sheet1.write(num+2+1,23,'E(H3)')
sheet1.write(num+2+1,24,'R_p(H3)')
sheet1.write(num+2+1,25,'E(He3)')
sheet1.write(num+2+1,26,'R_p(He3)')
sheet1.write(num+2+1,27,'E(He4)')
sheet1.write(num+2+1,28,'R_p(He4)')


sheet1.write(num+2+1,30,'Lambda-CCSD(T)(O16)')
sheet1.write(num+2+1,31,'Lambda-CCSD(T)(O24)')
sheet1.write(num+2+1,32,'Lambda-CCSD(T)(Ca40)')
sheet1.write(num+2+1,33,'Lambda-CCSD(T)(Ca48)')

sheet1.write(num+2+1,35,'R_p(O16)')
sheet1.write(num+2+1,36,'R_p(Ca40)')


num=num + 3
input_path = './set3_bad_Q_little_unbound/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set4_good_Q_little_larger_radii/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set5_bad_Q_little_unbound/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set6_bad_Q/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set7_bad_Q/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set8_bad_Q_best_others/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1



num = num + 1
input_path = './set9_good_Q/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set10_good_Q/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set11_good_Q/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set12_good_Q/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set13_good_Q/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set14_good_Q_best/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1

input_path = './set15_good_Q/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet1,num,input_path)
num = num + 1







#
# NNLO394
#

sheet2=filename.add_sheet("NNLO394")  



sheet2.write(1,0,'set_num')
sheet2.write(1,1,'c1')
sheet2.write(1,2,'c2')
sheet2.write(1,3,'c3')
sheet2.write(1,4,'c4')

sheet2.write(1,5,'Ct_1s0np')
sheet2.write(1,6,'Ct_1S0nn')
sheet2.write(1,7,'Ct_1s0pp')
sheet2.write(1,8,'Ct_3S1(pp,nn,np)')
sheet2.write(1,9,'C_1S0')
sheet2.write(1,10,'C_3P0')
sheet2.write(1,11,'C_3P1')
sheet2.write(1,12,'C_3P2')
sheet2.write(1,13,'C_1p1')
sheet2.write(1,14,'C_3S1')
sheet2.write(1,15,'C3S1-3D1')
sheet2.write(1,16,'cD')
sheet2.write(1,17,'cE')

sheet2.write(1,19,'E(H2)')
sheet2.write(1,20,'R_p(H2)')
sheet2.write(1,21,'Q(H2)')
sheet2.write(1,22,'P_D-state(H2)')
sheet2.write(1,23,'E(H3)')
sheet2.write(1,24,'R_p(H3)')
sheet2.write(1,25,'E(He3)')
sheet2.write(1,26,'R_p(He3)')
sheet2.write(1,27,'E(He4)')
sheet2.write(1,28,'R_p(He4)')


sheet2.write(1,30,'Lambda-CCSD(T)(O16)')
sheet2.write(1,31,'Lambda-CCSD(T)(O24)')
sheet2.write(1,32,'Lambda-CCSD(T)(Ca40)')
sheet2.write(1,33,'Lambda-CCSD(T)(Ca48)')

sheet2.write(1,35,'R_p(O16)')
sheet2.write(1,36,'R_p(Ca40)')

sheet2.write(1,38,'snm_E/A')
sheet2.write(1,39,'saturation_density')
sheet2.write(1,40,'pnm_E/A')
sheet2.write(1,41,'symmetric_energy')
num = 1
input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set20_new_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set21_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set22_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set23_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set24_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set25_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set26_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set27_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set28_Nmax14_best/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set29_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

input_path = '/home/slime/work/nulear_matter/plot_method/NNLO394/set30_Nmax14/'
LECs,few_body,BE,radii,two_plus,snm,dens,pnm,S_energy = read_file(input_path)
write_excel(filename,sheet2,num,input_path)
num = num + 1

















filename.save(output_path)  
