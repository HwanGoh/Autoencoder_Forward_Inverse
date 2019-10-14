#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""

import subprocess
from mpi4py import MPI
import copy
from schedule_and_run import get_hyperparameter_permutations, schedule_runs
from Training_Driver_Autoencoder_Fwd_Inv import HyperParameters
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class FLAGS:
    RECEIVED = 1
    RUN_FINISHED = 2
    EXIT = 3
    NEW_RUN = 4

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':
                    
    # To run this code "mpirun -n 5 ./Scheduler_Autoencoder_Fwd_Inv.py" in command line
    
    # mpi stuff
    comm   = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank   = comm.Get_rank()
    
    # By running "mpirun -n <number> ./scheduler.py", each process is cycled through by their rank
    if rank == 0: # This is the master processes' action 
        #########################
        #   Get Scenarios List  #
        #########################   
        hyper_p = HyperParameters() # Assign instance attributes below, DO NOT assign an instance attribute to GPU
        
        # assign instance attributes for hyper_p
        hyper_p.data_type         = ['full', 'bnd', 'bnd_only']
        hyper_p.num_hidden_layers = [3]
        hyper_p.truncation_layer  = [2] # Indexing includes input and output layer with input layer indexed by 0
        hyper_p.num_hidden_nodes  = [200]
        hyper_p.penalty           = [0.1, 0.5, 0.7]
        hyper_p.num_training_data = [20, 200, 2000]
        hyper_p.batch_size        = [20]
        hyper_p.num_epochs        = [4000]
        
        permutations_list, hyper_p_keys = get_hyperparameter_permutations(hyper_p) 
        print('permutations_list generated')
        
        # Convert each list in permutations_list into class attributes
        scenarios_class_instances = []
        for scenario_values in permutations_list: 
            hyper_p_scenario = HyperParameters()
            for i in range(0, len(scenario_values)):
                setattr(hyper_p_scenario, hyper_p_keys[i], scenario_values[i])
            scenarios_class_instances.append(copy.deepcopy(hyper_p_scenario))

        # Schedule and run processes
        schedule_runs(scenarios_class_instances, nprocs, comm)  
        
    else:  # This is the worker processes' action
        while True:
            status = MPI.Status()
            data = comm.recv(source=0, status=status)
            
            if status.tag == FLAGS.EXIT:
                break
            
            proc = subprocess.Popen(['./Training_Driver_Autoencoder_Fwd_Inv.py', f'{data.data_type}', f'{data.num_hidden_layers}', f'{data.truncation_layer}', f'{data.num_hidden_nodes}', f'{int(data.penalty)}', f'{data.num_training_data}', f'{data.batch_size}', f'{data.num_epochs}',  f'{data.gpu}'])
            proc.wait() # without this, the process will detach itself once the python code is done running
            
            req = comm.isend([], 0, FLAGS.RUN_FINISHED)
            req.wait() # without this, the message sent by comm.isend might get lost when this process hasn't been probed. With this, it essentially continues to message until its probe
    
    print('All scenarios computed')