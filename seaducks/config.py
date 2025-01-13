# seaducks/config.py
import numpy as np
config = {
    '81-10-9_random_states': [np.random.seed(val) for val in [32327, 45747, 52776, 63012, 74898, 95853, 96672, 108808, 113639, 126432]], 
    'return_variables' : ['lon', 'lat', 'id', 'time', 'drogue', 'u', 'v', 'Wx', 'Wy', 'Tx', 'Ty','u_raw', 'v_raw', 'Wx_raw', 'Wy_raw', 'Tx_raw', 'Ty_raw',
       'err_Wx', 'err_Wy', 'u_av', 'v_av', 'adt', 'sla', 'ugos', 'vgos',
       'adt_err', 'lon_var', 'lat_var','sst_x_derivative','sst_y_derivative']
}