import datetime

def run_experiment_for_all_configs(walltime:str,select:int,ncpus:int,mem:int,learing_rates:list,
                                   outfile_directory:str,array_indices:str,early_stopping:int,
                                   root_dir:str = "experiments/mvn_ngboost_fit_experiments/"):
    '''
    walltime: "hh:mm:ss"
    array_indices: "start-end" (inclusive)
    '''
    # initialising
    date = datetime.today().strftime('%Y-%m-%d')
    filename = f"run_experiment_date_{date}_eta"
    for lr in learing_rates:
        filename += f'_{lr}'
    with open (f'{filename}.pbs', 'w') as rsh:
        rsh.write('''\
        #! /bin/bash
        echo "I ran this"
        echo "more lines"
        ''')
    pass