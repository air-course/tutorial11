from scipy.spatial.transform import Rotation as R
from scipy import linalg
from scipy.special import gamma, factorial
import numpy as np

def direct_sphere(d,r_i=0,r_o=1):
    """Direct Sampling from the d Ball based on Krauth, Werner. Statistical Mechanics: Algorithms and Computations. Oxford Master Series in Physics 13. Oxford: Oxford University Press, 2006. page 42

    Parameters
    ----------
    d : int
        dimension of the ball
    r_i : int, optional
        inner radius, by default 0
    r_o : int, optional
        outer radius, by default 1

    Returns
    -------
    np.array
        random vector directly sampled from the solid d Ball
    """
    # vector of univariate gaussians:
    rand=np.random.normal(size=d)
    # get its euclidean distance:
    dist=np.linalg.norm(rand,ord=2)
    # divide by norm
    normed=rand/dist
    
    # sample the radius uniformly from 0 to 1 
    rad=np.random.uniform(r_i,r_o**d)**(1/d)
    # the r**d part was not there in the original implementation.
    # I added it in order to be able to change the radius of the sphere
    # multiply with vect and return
    return normed*rad


def sample_from_ellipsoid(M,rho,r_i=0,r_o=1):
    """sample directly from the ellipsoid defined by xT M x.

    Parameters
    ----------
    M : np.array
        Matrix M such that xT M x leq rho defines the hyperellipsoid to sample from
    rho : float
        rho such that xT M x leq rho defines the hyperellipsoid to sample from
    r_i : int, optional
        inner radius, by default 0
    r_o : int, optional
        outer radius, by default 1

    Returns
    -------
    np.array
        random vector from within the hyperellipsoid
    """
    lamb,eigV=np.linalg.eigh(M/rho) 
    d=len(M)
    xy=direct_sphere(d,r_i=r_i,r_o=r_o) #sample from outer shells
    T=np.linalg.inv(np.dot(np.diag(np.sqrt(lamb)),eigV.T)) #transform sphere to ellipsoid (refer to e.g. boyd lectures on linear algebra)
    return np.dot(T,xy.T).T

def najafi_based_sampling(
    plant, controller, n=10000, rho0=100, M=None, x_star=np.array([np.pi, 0])
):
    """Estimate the RoA for the closed loop dynamics using the method introduced in Najafi, E., Babuška, R. & Lopes, G.A.D. A fast sampling method for estimating the domain of attraction. Nonlinear Dyn 86, 823–834 (2016). https://doi.org/10.1007/s11071-016-2926-7

    Parameters
    ----------
    plant : simple_pendulum.model.pendulum_plant
        configured pendulum plant object
    controller : simple_pendulum.controllers.lqr.lqr_controller
        configured lqr controller object
    n : int, optional
        number of samples, by default 100000
    rho0 : int, optional
        initial estimate of rho, by default 10
    M : np.array, optional
        M, such that x_barT M x_bar is the Lyapunov fct. by default None, and controller.S is used
    x_star : np.array, optional
        nominal position (fixed point of the nonlinear dynamics)

    Returns
    -------
    rho : float
        estimated value of rho
    M : np.array
        M
    points: list containing all the points that were tested
    """

    rho = rho0

    points = []
    
    if M is None:
        M = np.array(controller.S)
    else:
        pass

    for i in range(n):
        # sample initial state from sublevel set
        # check if it fullfills Lyapunov conditions
        x_bar = sample_from_ellipsoid(M, rho)
        x = x_star + x_bar

        tau = controller.get_control_output([x[0], x[1]])

        xdot = plant.rhs(0, x, tau)

        V = x_bar.T @ M @ x_bar 

        Vdot = 2 * np.dot(x_bar, np.dot(M, xdot))

        if V > rho:
            print("something is fishy")
        # V < rho is true trivially, because we sample from the ellipsoid
        if Vdot > 0.0:  # if one of the lyapunov conditions is not satisfied
            rho = V

        points.append(x)
    
    return rho, M, points