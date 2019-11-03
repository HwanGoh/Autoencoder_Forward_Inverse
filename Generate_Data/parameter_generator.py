#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:37:51 2019

@author: hwan
"""

import numpy as np
import dolfin as dl
from gaussian_field import make_cov_chol

def ParameterGeneratorNineValues(V,solver,length = 0.8):
    chol = make_cov_chol(V, length)
    norm = np.random.randn(len(chol))
    generated_parameter = np.exp(0.5 * chol.T @ norm) 
    parameter_true_dl = ConvertArraytoDolfinFunction(V,generated_parameter)
    parameter_true_dl = solver.nine_param_to_function(solver.subfin_avg_op(parameter_true_dl))
    generated_parameter = parameter_true_dl.vector().get_local()
    return generated_parameter, parameter_true_dl

def ConvertArraytoDolfinFunction(V,nodal_values):
    nodal_values_dl = dl.Function(V)
    nodal_values_dl.vector().set_local(np.squeeze(nodal_values))
    return nodal_values_dl