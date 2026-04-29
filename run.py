import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import random 
import sympy as smp
import time
import os



from IPython.display import HTML

from util.pendulum_plant import PendulumPlant, plot_timeseries
from util.lqr import lqr, dlqr
from util.utilities_iLQR import check_type

mass = 0.06      # mass at the end of the pendulum [kg]
length = 0.1    # length of the rod [m]
damping = 0.0004   # viscious friction [kg m/s]
gravity = 9.81  # gravity [kg m²/s]
inertia = mass*length*length   # inertia of the pendulum [kg m²]
coulomb_fric = 0.0   # Coulomb friction coefficient [-]
torque_limit = np.inf  # torque limit of the motor, here we assume the motor can produce any torque
dt = 0.010            # time step size for iLQR (not simulations) [s]

pendulum = PendulumPlant(mass=mass,
                         length=length,
                         damping=damping,
                         gravity=gravity,
                         inertia=inertia,
                         torque_limit=torque_limit) 
                         
# The following is adapted from DFKI's iLQR code
# Large parts taken from `Russ Tedrake <https://github.com/RussTedrake/underactuated>`_.

class iLQR_Calculator():
    '''
    Class to calculate an optimal trajectory with an iterative
    linear quadratic regulator (iLQR). This implementation uses sympy.
    '''
    def __init__(self, n_x=2, n_u=1):
        """
        Class to calculate an optimal trajectory with an iterative
        linear quadratic regulator (iLQR). This implementation uses sympy.

        Parameters
        ----------
        n_x : int, default=2
            The size of the state space.
        n_u : int, default=1
            The size of the control space.
        """
        self.n_x = n_x
        self.n_u = n_u

        self.x_sym = smp.symbols("x:"+str(self.n_x))
        self.u_sym = smp.symbols("u:"+str(self.n_u))

    def set_start(self, x0):
        """
        Set the start state for the trajectory.

        Parameters
        ----------
        x0 : array-like
            the start state. Should have the shape of (n_x,)
        """
        self.x0 = x0

    def set_discrete_dynamics(self, dynamics_func):
        '''
        Sets the dynamics function for the iLQR calculation.

        Parameters
        ----------
        danamics_func : function
            dynamics_func should be a function with inputs (x, u) and output xd
        '''
        self.discrete_dynamics = dynamics_func

    def set_stage_cost(self, stage_cost_func):
        '''
        Set the stage cost (running cost) for the ilqr optimization.

        Parameters
        ----------
        stage_cost_func : function
            stage_cost_func should be a function with inputs (x, u)
            and output cost
        '''
        self.stage_cost = stage_cost_func

    def set_final_cost(self, final_cost_func):
        '''
        Set the final cost for the ilqr optimization.

        Parameters
        ----------
        final_cost_func : function
            final_cost_func should be a function with inputs x
            and output cost
        '''
        self.final_cost = final_cost_func

    def cost_trj(self, x_trj, u_trj):
        total = 0.0
        ln = 0.0
        N = x_trj.shape[0]
        for i in range(len(x_trj)-1):
            ln = ln + self.stage_cost(x_trj[i, :], u_trj[i, :]) / N
        lf = self.final_cost(x_trj[i+1, :])
        total = ln + lf
        return total
        
def pendulum_continuous_dynamics(x, u):
    md = check_type(x)

    m = mass
    l = length
    b = damping
    cf = coulomb_fric
    g = gravity
    inertia = m*l*l
    
    pos = x[0]
    vel = x[1]
    if md == smp:
        accn = (u[0] - m * g * l * md.sin(pos) - b * vel -
                cf*md.atan(1e8*vel)*2/np.pi) / inertia
    elif md == np:
        accn = (u[0] - m * g * l * md.sin(pos) - b * vel -
                cf*md.arctan(1e8*vel)*2/np.pi) / inertia
    xd = np.array([vel, accn])
    return xd
    
def pendulum_discrete_dynamics_euler(x, u):
    md = check_type(x)

    x_d = pendulum_continuous_dynamics(x, u)
    x_next = x + x_d*dt
    if md == smp:
        x_next = tuple(x_next)
    return x_next

def pendulum_discrete_dynamics_RK4(x, u):
    md = check_type(x)
    
    k1 = pendulum_continuous_dynamics(x,u)
    k2 = pendulum_continuous_dynamics(x + 0.5*dt*k1,u)
    k3 = pendulum_continuous_dynamics(x + 0.5*dt*k2,u)
    k4 = pendulum_continuous_dynamics(x + dt*k3,u)
    x_d = (k1 + 2 * (k2 + k3) + k4) / 6.0
    x_next = x + x_d*dt
    if md == smp:
        x_next = tuple(x_next)
    return x_next
    
def rollout(self, u_trj):
    x_trj = np.zeros((u_trj.shape[0]+1, self.x0.shape[0]))
    x = self.x0
    i = 0
    x_trj[i, :] = x
    for u in u_trj:
        i = i+1
        x = self.discrete_dynamics(x, u)
        x_trj[i, :] = x
    return x_trj

iLQR_Calculator.rollout = rollout

def pendulum_swingup_stage_cost(x, u):
    goal=[np.pi, 0]
    Cu=3
    Cp=0.00002
    Cv=0.000004
    Cen=0.0
    m=mass
    l=length
    b=damping
    cf=coulomb_fric
    g=gravity
    
    md = check_type(x)

    eps = 1e-6
    c_pos = (x[0] - goal[0] + eps)**2.0
    c_vel = (x[1] - goal[1] + eps)**2.0
    c_control = u[0]**2.0
    #c_control = u**2.0
    en_g = 0.5*m*(l*goal[1])**2.0 + m*g*l*(1.0-md.cos(goal[0]))
    en = 0.5*m*(l*x[1])**2.0 + m*g*l*(1.0-md.cos(x[0]))
    c_en = (en-en_g+eps)**2.0
    return Cu*c_control + Cp*c_pos + Cv*c_vel + Cen*c_en

def pendulum_swingup_final_cost(x):
    goal=[np.pi, 0]
    Cp=0.1
    Cv=0.02
    Cen=0.0
    m=mass
    l=length
    b=damping
    cf=coulomb_fric
    g=gravity
    
    md = check_type(x)

    eps = 1e-6
    c_pos = (x[0] - goal[0] + eps)**2.0
    c_vel = (x[1] - goal[1] + eps)**2.0
    en_g = 0.5*m*(l*goal[1])**2.0 + m*g*l*(1.0-md.cos(goal[0]))
    en = 0.5*m*(l*x[1])**2.0 + m*g*l*(1.0-md.cos(x[0]))
    c_en = (en-en_g+eps)**2.0
    return Cp*c_pos + Cv*c_vel + Cen*c_en   
    
def init_derivatives(self):
    """
    Initialize the derivatives of the dynamics.
    """
    x = self.x_sym
    u = self.u_sym
    
    l = self.stage_cost(x, u)

    l_x = smp.Matrix([l]).jacobian(x)
    self.l_x = smp.lambdify([x, u], l_x, "numpy")

    l_u = smp.Matrix([l]).jacobian(u)
    self.l_u = smp.lambdify([x, u], l_u, "numpy")

    l_xx = smp.Matrix([l_x]).jacobian(x)
    self.l_xx = smp.lambdify([x, u], l_xx, "numpy")

    l_ux = smp.Matrix([l_u]).jacobian(x)
    self.l_ux = smp.lambdify([x, u], l_ux, "numpy")

    l_uu = smp.Matrix([l_u]).jacobian(u)
    self.l_uu = smp.lambdify([x, u], l_uu, "numpy")

    l_final = self.final_cost(x)

    l_final_x = smp.Matrix([l_final]).jacobian(x)
    self.l_final_x = smp.lambdify([x], l_final_x, "numpy")

    l_final_xx = smp.Matrix([l_final_x]).jacobian(x)
    self.l_final_xx = smp.lambdify([x], l_final_xx, "numpy")

    f = self.discrete_dynamics(x, u)
    #print(x)
    f_x = smp.Matrix([f]).jacobian(x)
    self.f_x = smp.lambdify([x, u], f_x, "numpy")

    f_u = smp.Matrix([f]).jacobian(u)
    self.f_u = smp.lambdify([x, u], f_u, "numpy")

def compute_stage_cost_derivatives(self, x, u):
    l_x = np.atleast_1d(np.squeeze(self.l_x(x, u)))
    l_u = np.atleast_1d(np.squeeze(self.l_u(x, u)))
    l_xx = np.atleast_2d(np.squeeze(self.l_xx(x, u)))
    l_ux = np.atleast_2d(np.squeeze(self.l_ux(x, u)))
    l_uu = np.atleast_2d(np.squeeze(self.l_uu(x, u)))
    f_x = np.atleast_2d(np.squeeze(self.f_x(x, u)))
    f_u = np.atleast_2d(np.squeeze(self.f_u(x, u))).T
    return l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u

def compute_final_cost_derivatives(self, x):
    l_final_x = np.atleast_1d(np.squeeze(self.l_final_x(x)))
    l_final_xx = np.atleast_2d(np.squeeze(self.l_final_xx(x)))
    return l_final_x, l_final_xx

iLQR_Calculator.init_derivatives = init_derivatives
iLQR_Calculator.compute_stage_cost_derivatives = compute_stage_cost_derivatives
iLQR_Calculator.compute_final_cost_derivatives = compute_final_cost_derivatives

def Q_terms(self, l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx):
    Q_x = np.matmul(f_x.T, V_x) + l_x
    Q_u = np.matmul(f_u.T, V_x) + l_u
    Q_xx = l_xx + np.matmul(f_x.T, np.matmul(V_xx, f_x))
    Q_ux = l_ux + np.matmul(f_u.T, np.matmul(V_xx, f_x))
    Q_uu = l_uu + np.matmul(f_u.T, np.matmul(V_xx, f_u))
    return Q_x, Q_u, Q_xx, Q_ux, Q_uu

iLQR_Calculator.Q_terms = Q_terms

def gains(self, Q_uu, Q_u, Q_ux):
    #Q_uu_inv = np.linalg.inv(Q_uu + Q_uu.T) # why the addition of Quu transp.??
    Q_uu_inv = np.linalg.inv(Q_uu)
    #k = -2*np.matmul(Q_uu_inv, Q_u)
    #K = -2*np.matmul(Q_uu_inv, Q_ux) #multiplier?
    k = -np.matmul(Q_uu_inv, Q_u)
    K = -np.matmul(Q_uu_inv, Q_ux) #multiplier?

    k = np.atleast_1d(np.squeeze(k))
    K = np.atleast_2d(np.squeeze(K))
    return k, K

def V_terms(self, Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k):
    V_x = Q_x - np.matmul(K.T, np.matmul(Q_uu, k))
    V_xx = Q_xx - np.matmul(K.T, np.matmul(Q_uu, K))
    return V_x, V_xx

def expected_cost_reduction(self, Q_u, Q_uu, k):
    ecr = -Q_u.T*k - 0.5*k.T*(Q_uu*k)
    return np.squeeze(ecr)

iLQR_Calculator.gains = gains
iLQR_Calculator.V_terms = V_terms
iLQR_Calculator.expected_cost_reduction = expected_cost_reduction

def backward_pass(self, x_trj, u_trj, regu):
    k_trj = np.zeros([u_trj.shape[0], u_trj.shape[1]])
    K_trj = np.zeros([u_trj.shape[0], u_trj.shape[1], x_trj.shape[1]])
    expected_cost_redu = 0
    l_final_x, l_final_xx = self.compute_final_cost_derivatives(x_trj[-1])
    V_x = l_final_x
    V_xx = l_final_xx
    for n in range(u_trj.shape[0]-1, -1, -1):
        l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u = self.compute_stage_cost_derivatives(x_trj[n], u_trj[n])
        Q_x, Q_u, Q_xx, Q_ux, Q_uu = self.Q_terms(l_x, l_u, l_xx, l_ux, l_uu, f_x, f_u, V_x, V_xx)
        # We add regularization to ensure that Q_uu is invertible and nicely conditioned
        Q_uu_regu = Q_uu + np.eye(Q_uu.shape[0])*regu
        k, K = self.gains(Q_uu_regu, Q_u, Q_ux)
        k_trj[n, :] = k
        K_trj[n, :, :] = K
        V_x, V_xx = self.V_terms(Q_x, Q_u, Q_xx, Q_ux, Q_uu, K, k)
        expected_cost_redu += self.expected_cost_reduction(Q_u, Q_uu, k)
    return k_trj, K_trj, expected_cost_redu

iLQR_Calculator.backward_pass = backward_pass

def forward_pass(self, x_trj, u_trj, k_trj, K_trj):
    x_trj_new = np.zeros(x_trj.shape)
    x_trj_new[0, :] = x_trj[0, :]
    u_trj_new = np.zeros(u_trj.shape)
    for n in range(u_trj.shape[0]):
        u_trj_new[n, :] = u_trj[n] + k_trj[n] \
                          + np.matmul(K_trj[n],
                                      ((x_trj_new[n] - x_trj[n])))
        x_trj_new[n+1, :] = self.discrete_dynamics(x_trj_new[n],
                                                   u_trj_new[n])
    return x_trj_new, u_trj_new

iLQR_Calculator.forward_pass = forward_pass

def run_ilqr(self, N=50, init_u_trj=None, init_x_trj=None, shift=False,
             max_iter=50, break_cost_redu=1e-6, regu_init=100):
    """
    Run the iLQR optimization and receive a optimal trajectory for the
    defined cost function.

    Parameters
    ----------
    N : int, default=50
        The number of waypoints for the trajectory
    init_u_trj : array-like, default=None
        initial guess for the control trajectory
        ignored if None
    init_x_trj : array_like, default=None
        initial guess for the state space trajectory
        ignored if None
    shift : bool, default=False
        whether to shift the initial guess trajectories by one entry
        (delete the first entry and duplicate the last entry)
    max_iter : int, default=50
        optimization iterations the alogrithm makes at every timestep
    break_cost_redu : float, default=1e-6
        cost at which the optimization breaks off early
    regu_init : float, default=100
       initialization value for the regularizer

    Returns
    -------
    x_trj : array-like
        state space trajectory
    u_trj : array-like
        control trajectory
    cost_trace : array-like
        trace of the cost development during the optimization
    regu_trace : array-like
        trace of the regularizer development during the optimization
    redu_ratio_trace : array-like
        trace of ratio of cost_reduction and expected cost reduction
         during the optimization
    redu_trace : array-like
        trace of the cost reduction development during the optimization
    """
    # First forward rollout
    
    if init_u_trj is not None:
        u_trj = init_u_trj
        if shift:
            u_trj = np.delete(u_trj, 0, axis=0)
            u_trj = np.append(u_trj, [u_trj[-1]], axis=0)
    else:
        u_trj = np.random.randn(N-1, self.n_u)*0.0001

    if init_x_trj is not None:
        x_trj = init_x_trj
        if shift:
            x_trj = np.delete(x_trj, 0, axis=0)
            x_trj[0,:] = self.x0
            x_trj = np.append(x_trj,
                              [self.discrete_dynamics(x_trj[-1],
                               np.array(u_trj[-1]))], axis=0)
    else:
        x_trj = self.rollout(u_trj)

    total_cost = self.cost_trj(x_trj, u_trj)
    regu = regu_init
    max_regu = 10000
    min_regu = 0.01

    # Setup traces
    cost_trace = [total_cost]
    redu_ratio_trace = [1]
    redu_trace = []
    regu_trace = [regu]

    # Run main loop
    for it in range(max_iter):
        # Backward and forward pass
        k_trj, K_trj, expected_cost_redu = self.backward_pass(x_trj,
                                                              u_trj,
                                                              regu)
        x_trj_new, u_trj_new = self.forward_pass(x_trj, u_trj,
                                                 k_trj, K_trj)
        # Evaluate new trajectory
        total_cost = self.cost_trj(x_trj_new, u_trj_new)
        cost_redu = cost_trace[-1] - total_cost
        redu_ratio = cost_redu / abs(expected_cost_redu)
        # Accept or reject iteration
        if cost_redu > 0:
            # Improvement! Accept new trajectories and lower regularization
            redu_ratio_trace.append(redu_ratio)
            cost_trace.append(total_cost)
            x_trj = x_trj_new
            u_trj = u_trj_new
            regu *= 0.7
        else:
            # Reject new trajectories and increase regularization
            regu *= 2.0
            cost_trace.append(cost_trace[-1])
            redu_ratio_trace.append(0)
            if it > 10:
                print('over')
        regu = min(max(regu, min_regu), max_regu)
        regu_trace.append(regu)
        redu_trace.append(cost_redu)

        # Early termination if expected improvement is small
        if expected_cost_redu <= break_cost_redu or (cost_trace[-2]-cost_trace[-1]) < 1e-6:
            break

    return x_trj, u_trj, cost_trace, regu_trace, redu_ratio_trace, redu_trace, k_trj, K_trj

iLQR_Calculator.run_ilqr = run_ilqr

class iLQRController():
    def __init__(self, mass, length, damping, torque_limit, gravity, n_x = 2, n_u = 1, dt = 0.01):
        self.m = mass
        self.l = length
        self.g = gravity
        self.b = damping
        self.torque_limit = torque_limit
        self.i = 0
        self.iLQR = iLQR_Calculator(n_x = n_x, n_u = n_u)

    def set_discrete_dynamics(self,dyn_func):
        self.iLQR.set_discrete_dynamics(dyn_func)

    def set_stage_cost(self,s_cost):
        self.iLQR.set_stage_cost(s_cost)

    def set_final_cost(self,f_cost):
        self.iLQR.set_final_cost(f_cost)

    def set_start(self, x0):
        self.iLQR.set_start(x0)

    def run_ilqr(self, N, init_u_trj, init_x_trj, max_iter, regu_init, break_cost_redu=1e-6):
        self.iLQR.init_derivatives()
        self.x_trj, self.u_trj, self.cost_trace, self.regu_trace, self.redu_ratio_trace, self.redu_trace, self.k_trj, self.K_trj = self.iLQR.run_ilqr(N, init_u_trj, init_x_trj, max_iter, regu_init, break_cost_redu)
        self.N = N
    
    def restart_trajectory(self):
        self.i = 0
        
    def get_control_output(self,x):
        # PD controller is used to enforce the trajectory calculated by the iLQR
        
        if self.i>self.N-1.5:
            #print("REACHED END OF TRAJECTORY")
            tau = 0.0
        else:

            u = self.u_trj[self.i][0] + self.k_trj[self.i] \
                          + np.matmul(self.K_trj[self.i],
                                      ((x - self.x_trj[self.i])))
            u = u.item(0)
            tau = np.clip(u, -self.torque_limit, self.torque_limit)
            self.i += 1

        return tau

ilqr_controller = iLQRController(mass=mass, length=length, damping = damping, torque_limit=0.03, gravity=gravity)
ilqr_controller.set_discrete_dynamics(pendulum_discrete_dynamics_RK4)
ilqr_controller.set_stage_cost(pendulum_swingup_stage_cost)
ilqr_controller.set_final_cost(pendulum_swingup_final_cost)

x0 = np.array([0.001, 0.0])
ilqr_controller.set_start(x0)

ilqr_controller.run_ilqr(N = 1000, init_u_trj=None, init_x_trj=None, max_iter=100, regu_init=100, break_cost_redu=1e-6)


ilqr_controller.restart_trajectory()
tf = 10 # Final time (s)
dt = 0.01 # Time step (s)

print("running on hardware")

Treal_es, Xreal_es, Ureal_es, Ureal_es_des = pendulum.run_on_hardware_phys(tf, dt, controller=ilqr_controller) 


