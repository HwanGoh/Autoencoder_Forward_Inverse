#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 20:42:39 2019

@author: hwan
"""

import tensorflow as tf
import dolfin as dl
import numpy as np
import pandas as pd
import matplotlib as plt
from Thermal_Fin_Heat_Simulator.Utilities.gaussian_field import make_cov_chol
from Thermal_Fin_Heat_Simulator.Utilities.forward_solve import Fin
from Thermal_Fin_Heat_Simulator.Utilities.thermal_fin import get_space_2D, get_space_3D

###############################################################################
#                                   Loss                                      #
###############################################################################
def loss_model_augmented(autoencoder_pred, parameter_true):
    return tf.norm(tf.subtract(parameter_true, autoencoder_pred), 2, axis = 1)