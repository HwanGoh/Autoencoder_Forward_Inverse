import numpy as np

###############################################################################
#                                  Exponential                                #
###############################################################################
def exponential(parameter, measurement_points,
                parameter_dimensions, num_measurement_points):

    #=== Output ===#
    measurements = parameter[0]*np.exp(-parameter[1]*measurement_points.flatten())

    #=== Jacobian ===#
    Jac_forward = np.zeros((num_measurement_points, parameter_dimensions))
    for n in range(parameter.shape[0]):
        Jac_forward[:,0] = np.exp(-parameter[1]*measurement_points.flatten())
        Jac_forward[:,1] = -parameter[0]*measurement_points.flatten()*\
                np.exp(-parameter[1]*measurement_points.flatten())

    return measurements, Jac_forward
