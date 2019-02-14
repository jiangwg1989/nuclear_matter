import numpy as np

def charge_radii_calculator(r_point_square,N,Z):
    R_n_square = -0.1149
    R_p_square = 0.8775
    r_charge_square =  r_point_square + R_p_square + N/Z*R_n_square + 0.033
    return np.power(r_charge_square,0.5)


r_c = charge_radii_calculator(11.33480358   ,20,20)

print ('r_c='+str(r_c))

