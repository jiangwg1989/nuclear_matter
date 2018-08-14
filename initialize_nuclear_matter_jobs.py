#
# initialize the jobs for nuclear matter
#

import numpy as np
import os 



def output_ccm_in_file(file_path,cD,cE,particle_num,matter_type,density,nmax):
    with open(file_path,'w') as f_1:
        f_1.write('!Chiral order for Deltas(LO = 0,NLO=2,NNLO=3,N3LO=4) and cutoff'+'\n')
        f_1.write(str(3)+', '+str(450)+'\n')
        f_1.write('! cE and cD 3nf parameters:'+'\n' )
        f_1.write(str(cD)+', '+str(cE)+'\n')
        f_1.write('! number of particles'+'\n')
        f_1.write(str(particle_num)+'\n')
        f_1.write('! specify: pnm/snm, input type: density/kfermi'+'\n')
        f_1.write(matter_type+', density'+'\n')
        f_1.write('! specify boundary conditions (PBC/TABC/TABCsp)'+'\n')
        f_1.write('PBC'+'\n')
        f_1.write('! dens/kf, ntwist,  nmax'+'\n')
        f_1.write(str(density)+', '+'1'+', '+str(nmax)+'\n')
        f_1.write('! specify cluster approximation: CCD, CCDT'+'\n')
        f_1.write('CCD(T)'+'\n')
        f_1.write('! tnf switch (T/F) and specify 3nf approximation: 0=tnf0b, 1=tnf1b, 2=tnf2b'+'\n')
        f_1.write('T, 2'+'\n')
        f_1.write('! 3nf cutoff(MeV),non-local reg. exp'+'\n')
        f_1.write('450, 3'+'\n')


def output_job_file(file_path,cD,cE,particle_num,matter_type,density,nmax):
    with open(file_path,'w') as f_1:
        f_1.write('export OMP_NUM_THREADS=4'+'\n\n')
        f_1.write('/home/g1u/sw/intel_openmpi/bin/mpiexec -np '+'4'+' ./prog_ccm.exe  '+'ccm_in1'+' > '+'./output/'+str(matter_type)+str(particle_num)+'_cD_'+str(cD)+'_cE_'+str(cE)+'_rho_'+str(density)+'.out'+' &\n')
        f_1.write('wait'+'\n')

number = 1
file_path = './output/ccm_in'+str(number)
particle_num = 28
matter_type = 'snm'
nmax = 2

os.system('mkdir output')
output_ccm_in_file(file_path,-3,-1,particle_num,matter_type ,0.15,nmax)
file_path = './output/job.script'
output_job_file(file_path,-3,-1,particle_num,matter_type, 0.15,nmax)


