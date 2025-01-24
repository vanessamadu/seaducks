# seaducks/config.py
import numpy as np
import pickle

#rootdir = r'/rds/general/user/vm2218/home/phd-project1/SeaDucks/seaducks/'
rootdir = r"C:\Users\vm2218\OneDrive - Imperial College London\PhD Project\seaducks/"
with open(fr'{rootdir}81_10_9_seed_indices.p', 'rb') as pickle_file:
    seeds = pickle.load(pickle_file)

config = {
    '81-10-9_random_seeds':seeds,
    '81-10-9_random_states': [np.random.seed(val) for val in seeds], 
    'return_variables' : ['lon', 'lat', 'id', 'time', 'drogue', 'u', 'v', 'Wx', 'Wy', 'Tx', 'Ty','u_raw', 'v_raw', 'Wx_raw', 'Wy_raw', 'Tx_raw', 'Ty_raw',
       'err_Wx', 'err_Wy', 'u_av', 'v_av', 'adt', 'sla', 'ugos', 'vgos',
       'adt_err', 'lon_var', 'lat_var','sst_x_derivative','sst_y_derivative']
}