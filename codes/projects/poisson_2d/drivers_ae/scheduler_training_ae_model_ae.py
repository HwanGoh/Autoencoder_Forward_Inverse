#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""
import subprocess
from mpi4py import MPI
import copy

import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
from utils_scheduler.get_hyperparameter_permutations import get_hyperparameter_permutations
from utils_scheduler.schedule_and_run import schedule_runs

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class FLAGS:
    RECEIVED = 1
    RUN_FINISHED = 2
    EXIT = 3
    NEW_RUN = 4

###############################################################################
#                           Generate Scenarios List                           #
###############################################################################
def generate_scenarios_list():
    hyperp = {}
    hyperp['num_hidden_layers_encoder'] = [5]
    hyperp['num_hidden_layers_decoder'] = [2]
    hyperp['num_hidden_nodes_encoder']  = [1500]
    hyperp['num_hidden_nodes_decoder']  = [500]
    hyperp['activation']                = ['relu']
    hyperp['penalty_encoder']           = [10, 20, 30]
    hyperp['penalty_decoder']           = [10]
    hyperp['penalty_aug']               = [10]
    hyperp['penalty_prior']             = [0.001]
    hyperp['num_data_train']            = [10000]
    hyperp['batch_size']                = [100]
    hyperp['num_epochs']                = [1000]

    return get_hyperparameter_combinations(hyperp)

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':

    # To run this code "mpirun -n 5 ./scheduler_training_ae_model_aware.py" in command line

    # mpi stuff
    comm   = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank   = comm.Get_rank()

    # By running "mpirun -n <number> ./scheduler_", each
    # process is cycled through by their rank
    if rank == 0: # This is the master processes' action
        # Generate scenarios list
        scenarios_list = generate_scenarios_list()

        # Schedule and run processes
        schedule_runs(scenarios_list, nprocs, comm)

    else:  # This is the worker processes' action
        while True:
            status = MPI.Status()
            scenario = comm.recv(source=0, status=status)

            if status.tag == FLAGS.EXIT:
                break

            # Dump scenario to driver code and run
            scenario = json.dumps(scenario)
            proc = subprocess.Popen(['./training_driver_ae_model_aware.py',
                f'{scenario}'])
            # proc = subprocess.Popen(['./training_driver_ae_model_augmented_autodiff.py',
            #     f'{scenario}'])
            proc.wait()

            req = comm.isend([], 0, FLAGS.RUN_FINISHED)
            req.wait()

    print('All scenarios computed')
