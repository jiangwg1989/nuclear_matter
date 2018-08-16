######################################################
# processing the .out data 
######################################################

import numpy as np
import re
######################################################
# read .out data 
######################################################
def input_file(file_path):
    with open(file_path,'r') as f_1:
        count = len(open(file_path,'rU').readlines())
        data =  f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(0,count):
            if ( re.search('E/N', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                energy_per_nucleon = float(temp_1[2])
                return energy_per_nucleon
        print ('No "E/A" found in the file:'+file_path)

data_num = 585
raw_data = np.zeros((data_num,5),dtype = np.float)  # cD,cE,density,snm,pnm
#file_path = './output/output/pnm_14_cD_-3.00_cE_-1.00_rho_0.15.out'
#print (str(input_file(file_path)))
cE_min = -1
cE_max = 1
cE_gap = 0.25
cE_count = int( (cE_max - cE_min) / cE_gap + 1 )
cD_min = -3
cD_max = 3
cD_gap = 0.5
cD_count = int( (cD_max - cD_min) / cD_gap + 1 )
density_min = 0.15
density_max = 0.23
density_gap = 0.02
density_count = int( (density_max - density_min) / density_gap +1 )


######################################################
# read all the .out file for pnm 
######################################################
matter_type = 'pnm'
neutron_num = 14
loop4 = 0
for loop1 in range(cE_count):
    for loop2 in range(cD_count):
        cE = cE_min + cE_gap * loop1
        cD = cD_min + cD_gap * loop2
        for loop3 in range(density_count):
            density = density_min + density_gap * loop3
            ccm_out_file_path = './output/output/'+matter_type+'_%d_cD_%.2f_cE_%.2f_rho_%.2f.out' % (neutron_num,cD,cE,density)
            raw_data[loop4][4] = input_file(ccm_out_file_path) # 4 is the pnm
            loop4 = loop4 + 1
print ('total_count='+str(loop4))



######################################################
# read all the .out file for snm 
######################################################
matter_type = 'snm'
particle_num = 28
loop4 = 0
for loop1 in range(cE_count):
    for loop2 in range(cD_count):
        cE = cE_min + cE_gap * loop1
        cD = cD_min + cD_gap * loop2
        for loop3 in range(density_count):
            density = density_min + density_gap * loop3
            ccm_out_file_path = './output/output/'+matter_type+'_%d_cD_%.2f_cE_%.2f_rho_%.2f.out' % (particle_num,cD,cE,density)
            raw_data[loop4][0] = cD # 0 is the cD
            raw_data[loop4][1] = cE # 0 is the cE
            raw_data[loop4][2] = density # 3 is the density
            raw_data[loop4][3] = input_file(ccm_out_file_path) # 3 is the snm
            loop4 = loop4 + 1
if (data_num != loop4):
    print ('total_count error!!!')


######################################################
# write raw data file 
######################################################
file_path = './cD%.2f-%.2f_cE%.2f-%.2f.dat' % (cD_min,cD_max,cE_min,cE_max)
loop1 = 0
with open (file_path, 'w') as f:
    f.write(' cD    cE     density       E/A_snm       E/A_pnm \n')
    for loop1 in range(data_num):
        f.write('%.2f   %.2f    %.2f    %.8f    %.8f \n' % (raw_data[loop1][0],raw_data[loop1][1],raw_data[loop1][2],raw_data[loop1][3],raw_data[loop1][4]))


print (raw_data)
