import numpy as np

import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def continuous_linear(parameter, forward_vector):
    return np.dot(parameter, forward_vector)

def discrete_exponential(parameter, mesh, parameter_dimensions):

    #=== Output ===#
    state = parameter[0]*np.exp(-parameter[1]*mesh.flatten())

    #=== Jacobian ===#
    Jac_forward = np.zeros((len(mesh), parameter_dimensions))
    Jac_forward[:,0] = np.exp(-parameter[1]*mesh.flatten())
    Jac_forward[:,1] = -parameter[0]*mesh.flatten()*\
            np.exp(-parameter[1]*mesh.flatten())

    return state, Jac_forward
