####################################################
# initialize the jobs for nuclear matter
####################################################

import numpy as np
import os 



def output_ccm_in_file(file_path,cD,cE,particle_num,matter_type,density,nmax):
    with open(file_path,'w') as f_1:
        f_1.write('!Chiral order for Deltas(LO = 0,NLO=2,NNLO=3,N3LO=4) and cutoff'+'\n')
        f_1.write('3, 450\n')
        f_1.write('! cE and cD 3nf parameters:'+'\n' )
        f_1.write('%.2f, %.2f\n' % (cE,cD))
        f_1.write('! number of particles'+'\n')
        f_1.write('%d\n' % (particle_num) )
        f_1.write('! specify: pnm/snm, input type: density/kfermi'+'\n')
        f_1.write(matter_type+', density'+'\n')
        f_1.write('! specify boundary conditions (PBC/TABC/TABCsp)'+'\n')
        f_1.write('PBC'+'\n')
        f_1.write('! dens/kf, ntwist,  nmax'+'\n')
        f_1.write('%.2f, 1, %d\n' % (density, nmax))
        f_1.write('! specify cluster approximation: CCD, CCDT'+'\n')
        f_1.write('CCD(T)'+'\n')
        f_1.write('! tnf switch (T/F) and specify 3nf approximation: 0=tnf0b, 1=tnf1b, 2=tnf2b'+'\n')
        f_1.write('T, 2'+'\n')
        f_1.write('! 3nf cutoff(MeV),non-local reg. exp'+'\n')
        f_1.write('450, 3'+'\n')


#def output_job_file(file_path,cD,cE,particle_num,matter_type,density,nmax):
#    ccm_in_file_path = './output/ccm_in_'+matter_type+'_%d_cD_%.2f_cE_%.2f_rho_%.2f' % (particle_num,cD,cE,density)
#    ccm_out_file_path = './'+matter_type+'_%d_cD_%.2f_cE_%.2f_rho_%.2f.out &' % (particle_num,cD,cE,density)
#    with open(file_path,'w') as f_1:
#        f_1.write('export OMP_NUM_THREADS=4'+'\n\n')
#        f_1.write('/home/g1u/sw/intel_openmpi/bin/mpiexec -np 4 ./prog_ccm.exe '+ccm_in_file_path+' > '+ccm_out_file_path)
#        f_1.write('wait'+'\n')

number = 1
nmax = 2
os.system('mkdir output')
#output_ccm_in_file(file_path,-3,-1,particle_num,matter_type ,0.15,nmax)
#file_path = './output/job.script'
#output_job_file(file_path,-3,-1,particle_num,matter_type, 0.15,nmax)


####################################################
#  set up all the ccm_in files for snm 
####################################################
particle_num = 28
for loop1 in range(9):
    for loop2 in range(13):
        cD = -3 + loop2 * 0.5
        cE = -1 + loop1 * 0.25
        for loop3 in range(5):
            density = 0.15 + loop3 * 0.02
            #file_path = './output/ccm_in'+'snm'+str(particle_num)+'_cD_'+str('{:.2f}'.format(cD))+'_cE_'+str('{:.2f}'.format(cE))+'_rho_'+str(density)
            file_path = './output/ccm_in_snm_%d_cD_%.2f_cE_%.2f_rho_%.2f' % (particle_num,cD,cE,density) 
            output_ccm_in_file(file_path,cD,cE,particle_num,'snm',density,nmax)
            #print (file_path+'\n')
            #print ('cD,cE='+str(cD)+','+str(cE))
            #print (str(density))


####################################################
#  set up all the ccm_in files for pnm 
####################################################
neutron_num = 14 
for loop1 in range(9):
    for loop2 in range(13):
        cD = -3 + loop2 * 0.5
        cE = -1 + loop1 * 0.25
        for loop3 in range(5):
            density = 0.15 + loop3 * 0.02
            file_path = './output/ccm_in_pnm_%d_cD_%.2f_cE_%.2f_rho_%.2f' % (neutron_num,cD,cE,density) 
            output_ccm_in_file(file_path,cD,cE,neutron_num,'pnm',density,nmax)


 
####################################################
#  set up job script for snm 
####################################################
matter_type = 'snm'
particle_num = 28
file_path = './output/snm_job.script'
with open(file_path,'w') as f_1:
    f_1.write('export OMP_NUM_THREADS=16'+'\n\n')
    for loop1 in range(9):
        for loop2 in range(13):
            cD = -3 + loop2 * 0.5
            cE = -1 + loop1 * 0.25
            for loop3 in range(5):
                density = 0.15 + loop3 * 0.02
                ccm_in_file_path = './output/ccm_in_'+matter_type+'_%d_cD_%.2f_cE_%.2f_rho_%.2f' % (particle_num,cD,cE,density)
                ccm_out_file_path = './'+matter_type+'_%d_cD_%.2f_cE_%.2f_rho_%.2f.out &' % (particle_num,cD,cE,density)
                f_1.write('/home/g1u/sw/intel_openmpi/bin/mpiexec -np 4 ./prog_ccm.exe '+ccm_in_file_path+' > '+ccm_out_file_path+'\n')
                f_1.write('wait'+'\n')




####################################################
#  set up all the ccm_in files for pnm 
####################################################
matter_type = 'pnm'
neutron_num = 14
file_path = './output/pnm_job.script'
with open(file_path,'w') as f_1:
    f_1.write('export OMP_NUM_THREADS=16'+'\n\n')
    for loop1 in range(9):
        for loop2 in range(13):
            cD = -3 + loop2 * 0.5
            cE = -1 + loop1 * 0.25
            for loop3 in range(5):
                density = 0.15 + loop3 * 0.02
                ccm_in_file_path = './output/ccm_in_'+matter_type+'_%d_cD_%.2f_cE_%.2f_rho_%.2f' % (neutron_num,cD,cE,density)
                ccm_out_file_path = './'+matter_type+'_%d_cD_%.2f_cE_%.2f_rho_%.2f.out &' % (neutron_num,cD,cE,density)
                f_1.write('/home/g1u/sw/intel_openmpi/bin/mpiexec -np 4 ./prog_ccm.exe '+ccm_in_file_path+' > '+ccm_out_file_path+'\n')
                f_1.write('wait'+'\n')




