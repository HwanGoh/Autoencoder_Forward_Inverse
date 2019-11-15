#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 15:36:35 2019

@author: hwan
"""

from dolfin import *
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

mesh = UnitSquareMesh(5,5)
Q = VectorFunctionSpace(mesh, "CG", 1, dim=2)

v2d=vertex_to_dof_map(Q)
d2v=dof_to_vertex_map(Q)

pdb.set_trace()

v2d = v2d.reshape((-1, mesh.geometry().dim()))
d2v = d2v[range(0, len(d2v), 2)]/2

# Test
v = MeshFunction("size_t", mesh, mesh.topology().dim())
q = Function(Q)

v.array()[10] = 1.0
v.array()[25] = 1.0
q.vector()[v2d[10]] = 1.0
q.vector()[v2d[25]] = 1.0

plot(mesh)

#plot(v)
#plot(q)