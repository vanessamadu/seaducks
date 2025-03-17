from pbs_scripts import run_experiment_for_all_configs

# resources requested
walltime = "08:00:00"
select = 1
ncpus = 1
mem = 1
# experiment indexing
learning_rates = [0.1,1]
array_indices = '1-960'
early_stopping = 50


if __name__ == '__main__':
    run_experiment_for_all_configs(walltime,select,ncpus,mem,learning_rates,
                                    array_indices,early_stopping)