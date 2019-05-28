import numpy as np

def charge_radii_calculator(r_point_square,Z,N):
    R_n_square = -0.1149
    R_p_square = 0.77000625
    r_charge_square =  r_point_square + R_p_square + N*1.0/Z*R_n_square + 0.033
    print(N*1.0/Z)
    return np.power(r_charge_square,0.5)


r_c = charge_radii_calculator(11.90643787 ,20,34)

print ('r_c='+str(r_c))

