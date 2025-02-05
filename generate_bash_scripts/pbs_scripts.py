from datetime import datetime

def run_experiment_for_all_configs(walltime:str,select:int,ncpus:int,mem:int,learning_rates:list,
                                   array_indices:str,early_stopping:int,
                                   root_dir:str = "experiments/mvn_ngboost_fit_experiments"):
    '''
    walltime: "hh:mm:ss"
    array_indices: "start-end" (inclusive)
    '''
    # initialising
    date = datetime.today().strftime('%Y-%m-%d')
    filename = f"run_experiment_date_{date}_eta"
    for lr in learning_rates:
        filename += f'_{lr}'
    with open (f'{filename}.pbs', 'w') as rsh:
        rsh.write(f'''\
#! /bin/bash
#PBS -l walltime={walltime}
#PBS -l select={select}:ncpus={ncpus}:mem={mem}gb
#PBS -J {array_indices}

module load anaconda3/personal
eval "$(mamba shell hook --shell bash)"
export MAMBA_ROOT_PREFIX="$HOME/anaconda3"
mamba activate SeaDucks

cd $PBS_O_WORKDIR

# create directories that don't exist
mkdir -p {root_dir}/early_stopping_{early_stopping}/{date}/experiment_logs
mkdir -p {root_dir}/early_stopping_{early_stopping}/{date}/fit_models
mkdir -p {root_dir}/early_stopping_{early_stopping}/{date}/model_test_data
mkdir -p {root_dir}/early_stopping_{early_stopping}/{date}/out

python experiments/hpc_scripts/fit_mvn_ngboost.py ${{PBS_ARRAY_INDEX}} > experiment_${{PBS_ARRAY_INDEX}}_early_stopping_{early_stopping}_logs

# copy files back
mv experiment_${{PBS_ARRAY_INDEX}}_date_{date}_early_stopping_{early_stopping}_logs {root_dir}/early_stopping_{early_stopping}/{date}/experiment_logs
mv experiment_${{PBS_ARRAY_INDEX}}_date_{date}_early_stopping_{early_stopping}.p {root_dir}/early_stopping_{early_stopping}/{date}/fit_models
mv experiment_${{PBS_ARRAY_INDEX}}_date_{date}_early_stopping_{early_stopping}_test_data.p {root_dir}/early_stopping_{early_stopping}/{date}/model_test_data
mv {filename}.pbs.* {root_dir}/early_stopping_{early_stopping}/{date}/out
''')