import re
import numpy as np


def read_write_file(input_path,output_path):
    LECs = np.zeros(17)
    few_body = np.zeros(10)
    BE  = np.zeros(4)
    BE_flag = 0 
    radii = np.zeros(2)
    radii_flag = 0
    two_plus = np.zeros(2)
    two_plus_flag = 0
    with open(input_path,'r') as f_1:
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
                print(data[loop])
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
    with open(output_path,'w') as f_2:
        for loop1 in range(len(LECs)):
            f_2.write(str(LECs[loop1])+'  ')
        for loop1 in range(len(few_body)):
            f_2.write(str(few_body[loop1])+'  ')
        for loop1 in range(len(BE)):
            f_2.write(str(BE[loop1])+'  ')
        for loop1 in range(len(radii)):
            f_2.write(str(radii[loop1])+'  ')
#        for loop1 in range(len(two_plus)):
#            f_2.write(str(two_plus[loop1])+'  ')



input_path = './set20_Nmax14/info.txt'
output_path= './summary.txt'

read_write_file(input_path,output_path)
