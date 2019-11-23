#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""

import subprocess
from mpi4py import MPI
import copy
from Utilities.get_hyperparameter_permutations import get_hyperparameter_permutations
from Utilities.schedule_and_run import schedule_runs
from Training_Driver_Autoencoder_Fwd_Inv import Hyperparameters
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
                    
    # To run this code "mpirun -n 5 ./Scheduler_Training_Autoencoder_Fwd_Inv.py" in command line
    
    # mpi stuff
    comm   = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank   = comm.Get_rank()
    
    # By running "mpirun -n <number> ./Scheduler_Training_Autoencoder_Fwd_Inv.py", each process is cycled through by their rank
    if rank == 0: # This is the master processes' action 
        #########################
        #   Get Scenarios List  #
        #########################   
        hyperp = Hyperparameters() # Assign instance attributes below, DO NOT assign an instance attribute to GPU
        
        # assign instance attributes for hyperp
        hyperp.data_type         = ['full', 'bnd']
        hyperp.num_hidden_layers = [3]
        hyperp.truncation_layer  = [2] # Indexing includes input and output layer with input layer indexed by 0
        hyperp.num_hidden_nodes  = [50, 100, 500, 1000]
        hyperp.penalty           = [50]
        hyperp.batch_size        = [1000]
        hyperp.num_epochs        = [500]
        
        permutations_list, hyperp_keys = get_hyperparameter_permutations(hyperp) 
        print('permutations_list generated')
        
        # Convert each list in permutations_list into class attributes
        scenarios_class_instances = []
        for scenario_values in permutations_list: 
            hyperp_scenario = Hyperparameters()
            setattr(hyperp_scenario, 'which_gpu', '0') # Set gpu number as hyperp attribute so that MPI can pass it around with the other hyperparameters
            for i in range(0, len(scenario_values)):
                setattr(hyperp_scenario, hyperp_keys[i], scenario_values[i])
            scenarios_class_instances.append(copy.deepcopy(hyperp_scenario))

        # Schedule and run processes
        schedule_runs(scenarios_class_instances, nprocs, comm)  
        
    else:  # This is the worker processes' action
        while True:
            status = MPI.Status()
            data = comm.recv(source=0, status=status)
            
            if status.tag == FLAGS.EXIT:
                break
            
            proc = subprocess.Popen(['./Training_Driver_Autoencoder_Fwd_Inv.py', f'{data.data_type}', f'{data.num_hidden_layers}', f'{data.truncation_layer}', f'{data.num_hidden_nodes}', f'{data.penalty:.4f}', f'{data.batch_size}', f'{data.num_epochs}',  f'{data.which_gpu}'])
            proc.wait() # without this, the process will detach itself once the python code is done running
            
            req = comm.isend([], 0, FLAGS.RUN_FINISHED)
            req.wait() # without this, the message sent by comm.isend might get lost when this process hasn't been probed. With this, it essentially continues to message until its probe
    
    print('All scenarios computed')