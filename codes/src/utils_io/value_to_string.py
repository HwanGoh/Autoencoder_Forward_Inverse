#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  28 10:16:28 2020

@author: hwan
"""
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

def value_to_string(value):
    if value >= 1:
        value = int(value)
        string = str(value)
    else:
        value_decimal_form = '%.9f'%(value)
        string = 'pt'

        found_nonzero_value_flag = 0
        for n in range(2,9):
            if found_nonzero_value_flag == 1:
                if value_decimal_form[n] == '0': # If found a zero
                    break
            if value_decimal_form[n] == '0':
                string += '0'
            else:
                string += value_decimal_form[n]
                found_nonzero_value_flag = 1

        if found_nonzero_value_flag != 1: # If no nonzeros found
            string = '0'

    return string
