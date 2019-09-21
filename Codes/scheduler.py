#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:53:06 2019

@author: Jon Wittmer
"""

import nvidia_smi
import copy
import subprocess
from mpi4py import MPI
from time import sleep
from Training_Driver_Autoencoder_Fwd_Inv import HyperParameters
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"


class FLAGS:
    RECEIVED = 1
    RUN_FINISHED = 2
    EXIT = 3
    NEW_RUN = 4

###############################################################################
#                        Generate List of Scenarios                           #
###############################################################################
def get_scenarios_list(hyper_p):
    hyper_p_list = [hyper_p.num_hidden_layers, hyper_p.truncation_layer,  hyper_p.num_hidden_nodes, \
                    hyper_p.penalty, hyper_p.num_training_data, hyper_p.batch_size, hyper_p.num_epochs]
    
    scenarios_list = assemble_parameters(hyper_p_list)
        
    scenarios = []
    for vals in scenarios_list:
        hyper_p                   = HyperParameters()
        hyper_p.num_hidden_layers = vals[0]
        hyper_p.truncation_layer  = vals[1]
        hyper_p.num_hidden_nodes  = vals[2]
        hyper_p.penalty           = vals[3]
        hyper_p.num_training_data = vals[4]
        hyper_p.batch_size        = vals[5]
        hyper_p.num_epochs        = vals[6]
        
        scenarios.append(copy.deepcopy(hyper_p))

    return scenarios

def assemble_parameters(hyper_p_list):
    # params is a list of lists, with each inner list representing
    # a different model parameter. This function constructs the combinations
    return get_combinations(hyper_p_list[0], hyper_p_list[1:])
    
def get_combinations(hyper_p, hyper_p_list):
    # assign here in case this is the last list item
    combos = hyper_p_list[0]
    
    # reassign when it is not the last item - recursive algorithm
    if len(hyper_p_list) > 1:
        combos = get_combinations(hyper_p_list[0], hyper_p_list[1:])
        
    # concatenate the output into a list of lists
    output = []
    for i in hyper_p:
        for j in combos:
            # convert to list if not already
            j = j if isinstance(j, list) else [j]            
            # for some reason, this needs broken into 3 lines...Python
            temp = [i]
            temp.extend(j)
            output.append(temp)
    return output                

###############################################################################
#                            Schedule and Run                                 #
###############################################################################
def schedule_runs(scenarios, nproc, comm, total_gpus = 4):
    
    nvidia_smi.nvmlInit()
    
    scenarios_left = len(scenarios)
    print(str(scenarios_left) + ' total runs left')
    
    # initialize available processes
    available_processes = list(range(1, nprocs))
    
    flags = FLAGS()
    
    # start running tasks
    while scenarios_left > 0:
        
        # check worker processes for returning processes
        s = MPI.Status()
        comm.Iprobe(status=s)
        if s.tag == flags.RUN_FINISHED:
            print('Run ended. Starting new thread.')
            data = comm.recv() 
            scenarios_left -= 1
            if len(scenarios) == 0:
                comm.send([], s.source, flags.EXIT)
            else: 
                available_processes.append(s.source) 

        # assign training to process
        available_gpus = available_GPUs(total_gpus) # check which GPUs have available memory or computation space

        if len(available_gpus) > 0 and len(available_processes) > 0 and len(scenarios) > 0:
            curr_process = available_processes.pop(0) # rank of the process to send to
            curr_scenario = scenarios.pop(0)
            curr_scenario.gpu = str(available_gpus.pop(0)) # which GPU we want to run the process on. Note that the extra "gpu" field is created here as well
            
            print('Beginning Training of NN:')
            print_scenario(curr_scenario)
            print()
            
            # block here to make sure the process starts before moving on so we don't overwrite buffer
            print('current process: ' + str(curr_process))
            req = comm.isend(curr_scenario, curr_process, flags.NEW_RUN)
            req.wait() # effectively makes it a synchronous call
            
        elif len(available_processes) > 0 and len(scenarios) == 0:
            while len(available_processes) > 0:
                proc = available_processes.pop(0) # removes all leftover processes in the event that al scenarios are complete
                comm.send([], proc, flags.EXIT)

        sleep(30) # Tensorflow environment takes a while to fill up the GPU. This sleep command gives tensorflow time to fill up the GPU before checking if its available       
    
def available_GPUs(total_gpus):
    available_gpus = []
    for i in range(total_gpus):
        handle  = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        res     = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        if res.gpu < 30 and (mem_res.used / mem_res.total *100) < 30: # Jon heuristically defines what it means for a GPU to be available
            available_gpus.append(i)
    return available_gpus

def print_scenario(p):
    print()
    print(f'    p.num_hidden_layers:   {p.num_hidden_layers}')
    print(f'    p.truncation_layer:    {p.truncation_layer}')
    print(f'    p.num_hidden_nodes:    {p.num_hidden_nodes}')
    print(f'    p.penalty:             {p.penalty}')
    print(f'    p.num_training_data:   {p.num_training_data}')
    print(f'    p.batch_size:          {p.batch_size}')
    print(f'    p.num_epochs:          {p.num_epochs}')
    print()

###############################################################################
#                                   Executor                                  #
###############################################################################
if __name__ == '__main__':
    
    # To run this code "mpirun -n <number> ./scheduler.py" in command line
    
    # mpi stuff
    comm   = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank   = comm.Get_rank()

    
    # By running "mpirun -n <number> ./scheduler.py", each process is cycled through by their rank
    if rank == 0: # This is the master processes' action 
        #########################
        #   Get Scenarios List  #
        #########################   
        hyper_p = HyperParameters()
            
        hyper_p.num_hidden_layers = [1]
        hyper_p.truncation_layer = [2] # Indexing includes input and output layer
        hyper_p.num_hidden_nodes = [200]
        hyper_p.penalty = [1, 10, 20, 30, 40]
        hyper_p.num_training_data = [20, 200, 2000]
        hyper_p.batch_size = [20]
        hyper_p.num_epochs = [50000]
        
        scenarios = get_scenarios_list(hyper_p)      
        schedule_runs(scenarios, nprocs, comm)  
    else:  # This is the worker processes' action
        while True:
            status = MPI.Status()
            data = comm.recv(source=0, status=status)
            
            if status.tag == FLAGS.EXIT:
                break
            
            proc = subprocess.Popen(['./Training_Driver_Autoencoder_Fwd_Inv.py', f'{data.num_hidden_layers}', f'{data.truncation_layer}', f'{data.num_hidden_nodes}', f'{int(data.penalty)}', f'{data.num_training_data}', f'{data.batch_size}', f'{data.num_epochs}',  f'{data.gpu}'])
            proc.wait()
            
            req = comm.isend([], 0, FLAGS.RUN_FINISHED)
            req.wait()
    
    print('All scenarios computed')