from dolfin import *
import numpy as np
from mshr import Rectangle, generate_mesh  
import pdb #Equivalent of keyboard in MATLAB, just add "pdb.set_trace()"

class SubFin(SubDomain):
    def __init__(self, subfin_bdry, **kwargs):
        self.y_b = subfin_bdry[0]
        self.is_left = subfin_bdry[1]
        super(SubFin, self).__init__(**kwargs)

    def inside(self, x, on_boundary):
        if self.is_left:
            return (between(x[1], (self.y_b, self.y_b+0.75)) and between(x[0], (0.0, 2.5)))
        else:
            return (between(x[1], (self.y_b, self.y_b+0.75)) and between(x[0], (3.5, 6.0)))

class SubFinBoundary(SubDomain):
    def __init__(self, subfin_bdry, **kwargs):
        self.y_b = subfin_bdry[0]
        self.is_left = subfin_bdry[1]
        super(SubFinBoundary, self).__init__(**kwargs)

    def inside(self, x, on_boundary):
        if self.is_left:
            return (on_boundary and between(x[1], (self.y_b, self.y_b+0.75)) 
                    and between(x[0], (0.0, 2.5)))
        else:
            return (on_boundary and between(x[1], (self.y_b, self.y_b+0.75)) 
                    and between(x[0], (3.5, 6.0)))

class CenterFin(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (2.5, 3.5))

class CenterFinBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return (on_boundary and between(x[0], (2.5, 3.5)) and not (near(x[1], 0.0)))

class SubfinExpr(UserExpression):
    def __init__(self, subfin_bdry, **kwargs):
        self.y_b = subfin_bdry[0]
        self.isLeft = subfin_bdry[1]
        super(SubfinExpr, self).__init__(**kwargs)

    def eval(self, value, x):
        y_t = self.y_b + 0.25
        if self.isLeft:
            if (x[1] >= self.y_b) and (x[1] <= y_t) and (x[0] < 2.5):
                value[0] = 1.0
            else:
                value[0] = 0.0
        else:
            if (x[1] >= self.y_b) and (x[1] <= y_t) and (x[0] > 3.5):
                value[0] = 1.0
            else:
                value[0] = 0.0

    def value_shape(self):
        return ()

class SubfinValExpr(UserExpression):
    def __init__(self, k_s, **kwargs):
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.k7, self.k8, self.k9 = k_s
        super(SubfinValExpr, self).__init__(**kwargs)
    def eval(self, value, x):
        if (x[0] >= 2.5) and (x[0] <= 3.5):
            value[0] = self.k5
        elif (x[0] < 2.5):
            if (x[1] >= 0.75) and (x[1] <= 1.0):
                value[0] = self.k1
            elif (x[1] >= 1.75) and (x[1] <= 2.0):
                value[0] = self.k2
            elif (x[1] >= 2.75) and (x[1] <= 3.0):
                value[0] = self.k3
            elif (x[1] >= 3.75) and (x[1] <= 4.0):
                value[0] = self.k4
            else:
                value[0] = 0.0
        else:
            if (x[1] >= 0.75) and (x[1] <= 1.0):
                value[0] = self.k9
            elif (x[1] >= 1.75) and (x[1] <= 2.0):
                value[0] = self.k8
            elif (x[1] >= 2.75) and (x[1] <= 3.0):
                value[0] = self.k7
            elif (x[1] >= 3.75) and (x[1] <= 4.0):
                value[0] = self.k6
            else:
                value[0] = 0.0
    def value_shape(self):
        return ()

class Fin:
    '''
    A class the implements the heat conduction problem for a thermal fin
    '''

    def __init__(self, V):
        '''
        Initializes a thermal fin instance for a given function space

        Arguments:
            V - dolfin FunctionSpace
        '''

        self.phi = None

        self.V = V
        self.dofs = len(V.dofmap().dofs()) 

        # Currently uses a fixed Biot number
        self.Bi = Constant(0.1)

        # Trial and test functions for the weak forms
        self.w = TrialFunction(V)
        self.v = TestFunction(V)

        self.w_hat = TestFunction(V)
        self.v_trial = TrialFunction(V)

        mesh = V.mesh()
        domains = MeshFunction("size_t", mesh, mesh.topology().dim())
        domains.set_all(0)

        self.fin1 = SubFin([0.75, True])
        self.fin2 = SubFin([1.75, True])
        self.fin3 = SubFin([2.75, True])
        self.fin4 = SubFin([3.75, True])
        self.fin5 = CenterFin()
        self.fin6 = SubFin([3.75, False])
        self.fin7 = SubFin([2.75, False])
        self.fin8 = SubFin([1.75, False])
        self.fin9 = SubFin([0.75, False])
        domains_sub = MeshFunction("size_t", mesh, mesh.topology().dim())
        domains_sub.set_all(0)
        self.fin1.mark(domains_sub, 1)
        self.fin2.mark(domains_sub, 2)
        self.fin3.mark(domains_sub, 3)
        self.fin4.mark(domains_sub, 4)
        self.fin5.mark(domains_sub, 5)
        self.fin6.mark(domains_sub, 6)
        self.fin7.mark(domains_sub, 7)
        self.fin8.mark(domains_sub, 8)
        self.fin9.mark(domains_sub, 9)

        # Marking boundaries for boundary conditions
        self.bottom = CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)
        self.exterior = CompiledSubDomain("!near(x[1], side) && on_boundary", side = 0.0)
        self.boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        self.boundaries.set_all(0)
        self.exterior.mark(self.boundaries, 1)
        self.bottom.mark(self.boundaries, 2)
        
        self.dx = Measure('dx', domain=mesh, subdomain_data=domains)
        self.ds = Measure('ds', domain=mesh, subdomain_data=self.boundaries)
        self.dx_s = Measure('dx', domain=mesh, subdomain_data=domains_sub)

        self._k = Function(V)

        self._F = inner(self._k * grad(self.w), grad(self.v)) * self.dx(0) + \
            self.v * self.Bi * self.w * self.ds(1)
        self._a = self.v * self.ds(2)
        _, self.domain_measure = self.averaging_operator()

        self.fin1_A = assemble(Constant(1.0) * self.dx_s(1))
        self.fin2_A = assemble(Constant(1.0) * self.dx_s(2))
        self.fin3_A = assemble(Constant(1.0) * self.dx_s(3))
        self.fin4_A = assemble(Constant(1.0) * self.dx_s(4))
        self.fin5_A = assemble(Constant(1.0) * self.dx_s(5))
        self.fin6_A = assemble(Constant(1.0) * self.dx_s(6))
        self.fin7_A = assemble(Constant(1.0) * self.dx_s(7))
        self.fin8_A = assemble(Constant(1.0) * self.dx_s(8))
        self.fin9_A = assemble(Constant(1.0) * self.dx_s(9))

        self.B_obs = self.observation_operator()

        # Randomly sampling state vector for inverse problems
        #  self.n_samples = 3
        #  self.samp_idx = np.random.randint(0, self.dofs, self.n_samples)    

    def forward(self, k):
        '''
        Performs a forward solve to obtain temperature distribution
        given the conductivity field m and FunctionSpace V.
        This solve assumes Biot number to be a constant.
        Returns:
         z - Temperature field 
         y - Average temperature (quantity of interest)
         A - Mass matrix
         B - Discretized RHS
         C - Averaging operator
        '''

        z = Function(self.V)

        self._k.assign(k)
        solve(self._F == self._a, z) 
        y = assemble(z * self.dx)/self.domain_measure

        return z, y


    def averaging_operator(self):
        '''
        Returns an operator that when applied to a function in V, gives the average.
        '''
        v = TestFunction(self.V)
        d_omega_f = interpolate(Expression("1.0", degree=2), self.V)
        domain_integral = assemble(v * self.dx)
        domain_measure = assemble(d_omega_f * self.dx)
        C = domain_integral/domain_measure
        C = C[:]
        return C, domain_measure

    def qoi_operator(self, x):
        '''
        Returns the quantities of interest given the state variable
        '''
        #  average = assemble(z * self.dx)/self.domain_measure

        #  z_vec = z.vector()[:] #TODO: Very inefficient
        #  rand_sample = z_vec[self.samp_idx]

        #TODO: External surface sampling. Most physically realistic
        return self.subfin_avg_op(x)

    def subfin_avg_op(self, k):
        # Subfin averages
        fin1_avg = assemble(k * self.dx_s(1))/self.fin1_A 
        fin2_avg = assemble(k * self.dx_s(2))/self.fin2_A 
        fin3_avg = assemble(k * self.dx_s(3))/self.fin3_A 
        fin4_avg = assemble(k * self.dx_s(4))/self.fin4_A 
        fin5_avg = assemble(k * self.dx_s(5))/self.fin5_A
        fin6_avg = assemble(k * self.dx_s(6))/self.fin6_A 
        fin7_avg = assemble(k * self.dx_s(7))/self.fin7_A 
        fin8_avg = assemble(k * self.dx_s(8))/self.fin8_A 
        fin9_avg = assemble(k * self.dx_s(9))/self.fin9_A 
        subfin_avgs = np.array([fin1_avg, fin2_avg, fin3_avg, fin4_avg, fin5_avg, 
            fin6_avg, fin7_avg, fin8_avg, fin9_avg])
        #  print("Subfin averages: {}".format(subfin_avgs))
        return subfin_avgs

    def nine_param_to_function(self, k_s):
        '''
        Same as five_param_to_function but does not assume symmetry.
        '''
        return interpolate(SubfinValExpr(k_s, degree=1), self.V)

    def observation_operator(self):
        z = TestFunction(self.V)
        fin1_avg = assemble(z * self.dx_s(1))/self.fin1_A 
        fin2_avg = assemble(z * self.dx_s(2))/self.fin2_A 
        fin3_avg = assemble(z * self.dx_s(3))/self.fin3_A 
        fin4_avg = assemble(z * self.dx_s(4))/self.fin4_A 
        fin5_avg = assemble(z * self.dx_s(5))/self.fin5_A
        fin6_avg = assemble(z * self.dx_s(6))/self.fin6_A 
        fin7_avg = assemble(z * self.dx_s(7))/self.fin7_A 
        fin8_avg = assemble(z * self.dx_s(8))/self.fin8_A 
        fin9_avg = assemble(z * self.dx_s(9))/self.fin9_A 

        B = np.vstack((
            fin1_avg[:],
            fin2_avg[:],
            fin3_avg[:],
            fin4_avg[:],
            fin5_avg[:],
            fin6_avg[:],
            fin7_avg[:],
            fin8_avg[:],
            fin9_avg[:]))

        return B
