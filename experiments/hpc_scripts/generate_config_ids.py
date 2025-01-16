import pickle

eta = {0.01,0.1,1}
min_leaf_data = {1,16,32}
max_leaves = {8,16,32,64}
sst_flag = {True, False}
polar_flag = {True, False}

index = 0 
config_ids = {}

for ii in eta:
    for jj in min_leaf_data:
        for kk in max_leaves:
            for ll in sst_flag:
                for mm in polar_flag:
                    config_ids[index] = [ii,jj,kk,ll,mm]
                    index +=1

filehandler = open(f"{'module_configuration_ids'}.p","wb")
pickle.dump(config_ids,filehandler)

print(config_ids)