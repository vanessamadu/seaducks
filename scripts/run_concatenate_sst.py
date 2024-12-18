import glob
#import os
import xarray as xr
import numpy as np
import pandas as pd

file_directory = r"D:\PhD\ocean-datasets\copernicus-data"
file_paths = sorted(glob.glob(f"{file_directory}/CMEMS_West_NA_sst_*.nc"))
output_filename = "Met_Office_West_NA_sst_1993_2019.nc"

# load datasets

datasets = [xr.open_dataset(file) for file in file_paths]
combined_dataset = xr.concat(datasets, dim='time')
output_filename = "Met_Office_West_NA_sst_1993_2019_full.nc"
# Save the combined dataset to a new NetCDF file
combined_dataset.to_netcdf(f"{output_filename}")

'''
datasets = [xr.open_dataset(file) for file in file_paths]

# save first dataset to output file
# Save the combined dataset to a new NetCDF file

# reindex to include leap years by adding the 29th February 1860 to each non-leap year dataset
all_years = np.arange(1993,2020)

# reindex 1993
new_index = pd.DatetimeIndex.insert(datasets[0].indexes['time'],59,np.datetime64('1860-02-29 00:00:00'))
datasets[0] = datasets[0].reindex(time=new_index)
print('reindex successful')
datasets[0].to_netcdf(f"{output_filename}",encoding ={"time": {"dtype": "int64"}})

for idx,dataset in enumerate(datasets):
    
    if idx > 0:
        if all_years[idx]%4 !=0:
            #reindex non-leap years
            new_index = pd.DatetimeIndex.insert(dataset.indexes['time'],59,np.datetime64('1860-02-29 00:00:00'))
            dataset = dataset.reindex(time=new_index)
            print(f"{all_years[idx]} reindexed successfully")
        dataset.to_netcdf(f"{output_filename}",mode='a',encoding ={"time": {"dtype": "int64"}})
        print(f"Dataset {idx} added to file")
    else:
        print(f"Dataset {idx} added to file")


#for file_path in file_paths:
#    os.remove(file_path)'''