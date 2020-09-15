import os
import json
import warnings

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def command_line_json_string_to_dict(args_list, hyperp):
    '''
    Overwrite the hyperparameters loaded from file.

    Note that the there is no checking to ensure that the command line
    arguments are in fact hyperparameters used by code, so spelling mistakes
    will cause the hyperparameters loaded from file to be used rather than
    the hyperparameter from command line. This was done on purpose to enable
    the capability to create new hyperparameters without changing hard-coding
    expected hyperparameters. This is more flexible, but more error prone.
    Warning added to notify user if key is not already present in hp dictionary.

    Assumes that all of the hyperparameters from the command line
    are in the form of a single JSON string in args[1]
    '''
    #=== Overwrite Hyperparameter Keys ===#
    command_line_arguments = json.loads(args_list[1])
    for key, value in command_line_arguments.items():
        if key not in hyperp:
            warnings.warn(
                f'Key "{key}" is not in hyper_p and has been added. Make sure this is correct.')
        hyperp[key] = value
    return hyperp
