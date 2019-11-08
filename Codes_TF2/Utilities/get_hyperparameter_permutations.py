#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 17:14:40 2019

@author: hwan
"""

###############################################################################
#                        Generate List of Scenarios                           #
###############################################################################
def get_hyperparameter_permutations(hyper_p):
    
    # Create list of hyperparameters and list of scenarios
    hyper_p_dict = hyper_p.__dict__ # converts all instance attributes into dictionary. Note that it does not include class attributes! In our case, this is GPU
    hyper_p_keys = list(hyper_p_dict.keys())
    hyper_p_dict_list = list(hyper_p_dict.values())           
    permutations_list = assemble_permutations(hyper_p_dict_list) # list of lists containing all permutations of the hyperparameters

    return permutations_list, hyper_p_keys

def assemble_permutations(hyper_p_dict_list):
    # params is a list of lists, with each inner list representing
    # a different model parameter. This function constructs the combinations
    return get_combinations(hyper_p_dict_list[0], hyper_p_dict_list[1:])
    
def get_combinations(hyper_p, hyper_p_dict_list):
    # assign here in case this is the last list item
    combos = hyper_p_dict_list[0]
    
    # reassign when it is not the last item - recursive algorithm
    if len(hyper_p_dict_list) > 1:
        combos = get_combinations(hyper_p_dict_list[0], hyper_p_dict_list[1:])
        
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