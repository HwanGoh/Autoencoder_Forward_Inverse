#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""
import socket
import subprocess
from mpi4py import MPI

import os
import sys
sys.path.insert(0, os.path.realpath('../../../src'))
import json

from utils_scheduler.get_hyperparameter_combinations import get_hyperparameter_combinations
from utils_scheduler.schedule_and_run_cpu import schedule_runs

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
    hyperp['num_hidden_nodes_encoder']  = [500]
    hyperp['num_hidden_nodes_decoder']  = [500]
    hyperp['activation']                = ['relu']
    hyperp['penalty_encoder']           = [10, 50, 100, 1000]
    hyperp['penalty_decoder']           = [10]
    hyperp['penalty_aug']               = [10, 50, 100, 1000]
    hyperp['penalty_prior']             = [0.1]
    hyperp['num_data_train']            = [10000]
    hyperp['batch_size']                = [100]
    hyperp['num_epochs']                = [1000]

    return get_hyperparameter_combinations(hyperp)

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':

    '''
    description:
        assigns one process to each cpu node. An alternative to this is multiple
        processes per cpu node.
    '''
    # mpi stuff
    comm   = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank   = comm.Get_rank()

    # By running "mpirun -n <number> ./scheduler.py", each process is cycled through by their rank
    if rank == 0: # This is the master processes' action
        # get scenarios list
        scenarios_list = generate_scenarios_list()

        # get the info for all processes
        processes = []
        while len(processes) < nprocs - 1:
            status = MPI.Status()
            proc_info = comm.recv()
            processes.append(proc_info)
        print(processes)

        # static gpu assignment per process. Currently only a single gpu per process
        nodes = {}
        active_procs = []
        for proc in processes:
            # keep track of the processes already found each node
            if proc['hostname'] not in nodes:
                nodes[proc['hostname']] = []
                nodes[proc['hostname']].append(str(proc['rank']))
                active_procs.append(proc['rank'])

        # Schedule and run processes
        schedule_runs(scenarios_list, active_procs, comm)

    else:
        # This is the worker processes' action
        # First send process info to master process
        # number of gpus in this node
        hostname = socket.gethostname()
        proc_info = {'rank': rank,
                     'hostname': hostname}
        req = comm.isend(proc_info, 0)
        req.wait()

        while True:
            status = MPI.Status()
            scenario = comm.recv(source=0, status=status)

            if status.tag == FLAGS.EXIT:
                break

            # convert dictionary to json
            scenario_json = json.dumps(scenario)
            # proc = subprocess.Popen(['./training_driver_ae.py',
            #     f'{scenario_json}',f'{"cpu"}'])
            proc = subprocess.Popen(['./training_driver_ae_model_augmented_autodiff.py',
                f'{scenario_json}',f'{"cpu"}'])
            proc.wait()

            req = comm.isend([], 0, FLAGS.RUN_FINISHED)
            req.wait()

    print('All scenarios computed')
