# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes
# pylint: disable=attribute-defined-outside-init

"""Copyright 2015 Roger R Labbe Jr.

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from copy import deepcopy
import numpy as np
from numpy import array, zeros, eye, dot
#from numpy.random import multivariate_normal
from filterpy.common import pretty_str, outer_product_sum
from decimal import Decimal 
from joblib import Parallel, delayed



class EnsembleKalmanFilter(object):
    """
    This implements the ensemble Kalman filter (EnKF). The EnKF uses
    an ensemble of hundreds to thousands of state vectors that are randomly
    sampled around the estimate, and adds perturbations at each update and
    predict step. It is useful for extremely large systems such as found
    in hydrophysics. As such, this class is admittedly a toy as it is far
    too slow with large N.

    There are many versions of this sort of this filter. This formulation is
    due to Crassidis and Junkins [1]. It works with both linear and nonlinear
    systems.

    Parameters
    ----------

    x : np.array(dim_x)
        state mean

    P : np.array((dim_x, dim_x))
        covariance of the state

    dim_z : int
        Number of of measurement inputs. For example, if the sensor
        provides you with position in (x,y), dim_z would be 2.

    dt : float
        time step in seconds

    N : int
        number of sigma points (ensembles). Must be greater than 1.

    K : np.array
        Kalman gain

    hx : function hx(x)
        Measurement function. May be linear or nonlinear - converts state
        x into a measurement. Return must be an np.array of the same
        dimensionality as the measurement vector.

    fx : function fx(x, dt)
        State transition function. May be linear or nonlinear. Projects
        state x into the next time period. Returns the projected state x.


    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        State estimate

    P : numpy.array(dim_x, dim_x)
        State covariance matrix

    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    z : numpy.array
        Last measurement used in update(). Read only.

    R : numpy.array(dim_z, dim_z)
        Measurement noise matrix

    Q : numpy.array(dim_x, dim_x)
        Process noise matrix

    fx : callable (x, dt)
        State transition function

    hx : callable (x)
        Measurement function. Convert state `x` into a measurement

    K : numpy.array(dim_x, dim_z)
        Kalman gain of the update step. Read only.

    inv : function, default numpy.linalg.inv
        If you prefer another inverse function, such as the Moore-Penrose
        pseudo inverse, set it to that instead: kf.inv = np.linalg.pinv

    Examples
    --------

    .. code-block:: Python

        def hx(x):
           return np.array([x[0]])

        F = np.array([[1., 1.],
                      [0., 1.]])
        def fx(x, dt):
            return np.dot(F, x)

        x = np.array([0., 1.])
        P = np.eye(2) * 100.
        dt = 0.1
        f = EnsembleKalmanFilter(x=x, P=P, dim_z=1, dt=dt,
                                 N=8, hx=hx, fx=fx)

        std_noise = 3.
        f.R *= std_noise**2
        f.Q = Q_discrete_white_noise(2, dt, .01)

        while True:
            z = read_sensor()
            f.predict()
            f.update(np.asarray([z]))

    See my book Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

    References
    ----------

    - [1] John L Crassidis and John L. Junkins. "Optimal Estimation of
      Dynamic Systems. CRC Press, second edition. 2012. pp, 257-9.
    """

    def __init__(self, x, P, dim_z, dt, N, hx, fx, inf=1, gauss_kick=None, ensemble_type="gsn", constraints=None, qscales=None, 
    inf_a=None, seed=None):
        if dim_z <= 0:
            raise ValueError('dim_z must be greater than zero')

        if N <= 0:
            raise ValueError('N must be greater than zero')

        dim_x = len(x)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.N = N
        self.hx = hx
        self.fx = fx
        self.K = zeros((dim_x, dim_z))
        self.z = array([[None] * self.dim_z]).T
        self.S = zeros((dim_z, dim_z))   # system uncertainty
        self.SI = zeros((dim_z, dim_z))  # inverse system uncertainty
        self.inf= inf
        self.ensemble_type = ensemble_type
        self.inf_a = inf_a 
        self._rng =  np.random.default_rng(seed)
        self.initialize(x, P)
        
        self.Q = eye(dim_x)       # process uncertainty
        self.R = eye(dim_z)       # state uncertainty
        self.inv = np.linalg.inv
        self.gauss_kick = gauss_kick
        self.constraints = constraints
        self.qscales = qscales
        # used to create error terms centered at 0 mean for
        # state and measurement
        
        self._mean = zeros(dim_x)
        self._mean_z = zeros(dim_z)
        
        ## validate some input
        assert self.inf_a!=0, "inf_a should be non-zero. If you do not want analysis inflation set inf_a=None"
        
        if self.qscales is not None:
            assert self.qscales.size==self.dim_x, "qscales must be dim_x size (zero entries can be specified)"
        
        if self.constraints is not None:
            assert self.constraints.shape[0] == 2, "constraints must be an array with 2 rows for lower and upper bounds respectively. "
            assert self.constraints.shape[-1] == self.dim_x, "constraints.shape[-1] must have dim_x dimensions"
                
            self._lb_idx = self.constraints[0,:].nonzero()
            self._ub_idx = self.constraints[1,:].nonzero() 
            
            self.constraints[0,self.constraints[0,:] == None] = -np.inf*np.ones(dim_x)[self.constraints[0,:] == None] #translate the None bounds into infinity 
            self.constraints[1,self.constraints[1,:] == None] = np.inf*np.ones(dim_x)[self.constraints[1,:] == None] 
                    
    def initialize(self, x, P):
        """
        Initializes the filter with the specified mean and
        covariance. Only need to call this if you are using the filter
        to filter more than one set of data; this is called by __init__

        Parameters
        ----------

        x : np.array(dim_z)
            state mean

        P : np.array((dim_x, dim_x))
            covariance of the state
        """

        if x.ndim != 1:
            raise ValueError('x must be a 1D array')
        
        if self.ensemble_type == "gsn":
            self.sigmas = self._rng.multivariate_normal(mean=x, cov=P, size=self.N)
        elif self.ensemble_type == "uniform":
            print("Uniform ensemble is legacy code from previous CCFE software")
            uniform_ensemble = np.zeros((self.N, self.dim_x))
            for i in range(self.dim_x):
                uniform_ensemble[:,i] = self._rng.uniform(high=P[i,i]+x[i], low=-P[i,i]+x[i], size=self.N)
            self.sigmas = uniform_ensemble
    
        #print(self.sigmas.shape)
        self.x = x
        self.P = P

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def update(self, z, R=None):
        """
        Add a new measurement (z) to the kalman filter. If z is None, nothing
        is changed.

        Parameters
        ----------

        z : np.array
            measurement for this update.

        R : np.array, scalar, or None
            Optionally provide R to override the measurement noise for this
            one call, otherwise self.R will be used.
        """
        #model error correction implemented by Luca
        if self.gauss_kick is not None:
            self.sigmas += self._rng.normal( np.zeros((self.N, self.dim_x)) , np.tile(self.gauss_kick , (self.N, self.dim_x) ) )
            self.x = np.mean(self.sigmas, axis=0)
        elif self.qscales is not None: #nonuniform version of Luca's model error correction
            self.sigmas+=self._rng.normal( np.zeros((self.N, self.dim_x)) , np.tile(self.qscales, (self.N, 1) ) )
            self.x = np.mean(self.sigmas, axis=0)
        
        if z is None:
            self.z = array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return
        
        if self.constraints is not None:
            self.constrain() 
            self.x = np.mean(self.sigmas, axis=0)
        
        self.sigma_prior = np.copy(self.sigmas)
        
        
        if R is None:
            R = self.R
        if np.isscalar(R):
            R = eye(self.dim_z) * R

        N = self.N
        dim_z = len(z)
        sigmas_h = zeros((N, dim_z))

        # transform sigma points into measurement space
        #inflation only works with a linear h in this definition
        for i in range(N):
            sigmas_h[i] = self.hx(self.sigmas[i])

        z_mean = np.mean(sigmas_h, axis=0)


        P_zz = (outer_product_sum(self.inf*(sigmas_h - z_mean)) / (N-1)) + R #as self.hx applies a linear transform we multiply by self.inf here
        P_xz = outer_product_sum(
            self.inf*self.sigmas - self.inf*self.x, self.inf*sigmas_h - self.inf*z_mean) / (N - 1) 

        self.S = P_zz
        #self.SI = self.inv(self.S)
        #self.K = dot(P_xz, self.SI)
        self.K = np.linalg.solve(self.S.T, P_xz.T).T #stable solve of the above problem
        e_r = self._rng.multivariate_normal(self._mean_z, R, N) #mean z is a vector of zeros

        
        for i in range(N):
            self.sigmas[i] += dot(self.K, z + e_r[i] - sigmas_h[i])
            
        if self.inf_a is not None: #this hasn't been validated
            self.sigmas = (1-self.inf_a)*self.sigmas + self.inf_a*self.sigma_prior
            
        if self.constraints is not None:
            self.constrain() 

                    
        self.x = np.mean(self.sigmas, axis=0)
        self.P = self.P - dot(dot(self.K, self.S), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_anaet(self, dt_interval):
        """
        Function to perform faster integration of the ANAET model over the assimilation intverval.
        This is NOT compatible with process noise added on integration time-steps. If you wish to use 
        process noise on integration time-steps you must use the normal predict attribute. Admittedly this is much 
        slower and you may need to develop another integration method in this instance 
        
        Note we are storing solutions from the next integration step (dt) and returning values up to the next
        assimilation forecast step (dt_interval). t=0 is NOT stored as this counts as the previous assimilation step. 
        
        Parameters
        ----------
        dt_interval : float
            Time to forecast model at for the next assimilation.
            

        Returns
        -------
        means : np.array (n_times, dim_x)
            Means of the enkf method between next integration step (dt) and the assimilation point (dt_interval).
        priors : np.array (n_times, dim_x, dim_x)
            Priors between dt and dt_interval.

        """
        if self.constraints is not None:
            raise ValueError("predict_anaet is not compatible with naive constraints")
        
        N = self.N
        
        
        #this section has to be included as sometimes np.arange lands close to the end point due to numerical rounding errors
        #depending on self.dt and dt_interval. Consider changing np.arange to list comprehension instead 
        #trange = np.array([self.dt*i for i in range(int(dt_interval/self.dt)+1)])
        
        ## This method allows for observation points that do not land exactly on integration time-steps (which is perfectly fine in enkf)
        trange=np.arange(0, dt_interval, self.dt)
        dtstr = Decimal(str(self.dt))
        nodecimals = np.abs(dtstr.as_tuple().exponent)
        trange=np.round(trange, nodecimals) #round to avoid numerical addition errors 
        n_times = trange.shape[0]  
        if trange[-1]!=dt_interval:
            trange=np.append(trange, dt_interval)
            
        n_times = int(dt_interval/self.dt) 
        solns = np.zeros((self.N, n_times, self.x.shape[0]))
        

        for i, s in enumerate(self.sigmas): #could possibly speed up by vectorising this bit of code (will not work with numbalsoda)
           solns[i,:,:] = self.fx(s, self.dt, dt_interval, trange)
           self.sigmas[i] = solns[i,-1,:]
        
        #process noise not compatible with this style of prediction step right now
        e = self._rng.multivariate_normal(self._mean, self.Q, N)
        self.sigmas += e
        
        
        self.x = np.mean(self.sigmas, axis=0)
        self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)
        self.P_prior = np.copy(self.P)
            
        
        # save prior
        self.x_prior = np.copy(self.x)      
        
        priors = np.zeros((n_times, self.x.shape[0], self.x.shape[0]))
        means = np.mean(solns, axis=0)
        
        #can calculate all priors but this is slow
        for i in range(n_times):
            priors[i, :, :] = outer_product_sum(solns[:,i,:] - means[i,:]) / (N - 1)

        return means, priors 
    

    
    def constrain(self):
        """
        Constrain the elements by upper and lower bounds. Note you only need to do this for nonzero bounds.
        (though currently it runs through them all)
    
        """
    
        for i, s in enumerate(self.sigmas):
            lb_violations = np.less_equal(s, self.constraints[0,:])
            self.sigmas[i][lb_violations] = self.constraints[0,lb_violations] 
            ub_violations = np.greater_equal(s, self.constraints[1,:])

            self.sigmas[i][ub_violations] = self.constraints[1,ub_violations]


    def predict(self, assim):
        """ Predict next position. """
        
        N = self.N

        """
        while the time is less than the assimilation time, 
        evolve the models forward in parallel 
        
        """
        for i, s in enumerate(self.sigmas): #could possibly speed up by vectorising this bit of code (hard to generalise for all functions)
            self.sigmas[i] = self.fx(s, self.dt)
        
        
        
        e = self._rng.multivariate_normal(self._mean, self.Q, N)
        if assim:
            self.sigmas += e
        
        #check the constraints are valid
        if self.constraints is not None:
            self.constrain()
        
        
        self.x = np.mean(self.sigmas, axis=0)
        
        #you can set assim to False if you only want to calculate these at assimilation times
        if assim==True:
            self.P = outer_product_sum(self.sigmas - self.x) / (N - 1)
            self.P_prior = np.copy(self.P)
            
        # save prior
        self.x_prior = np.copy(self.x)


    def __repr__(self):
        return '\n'.join([
            'EnsembleKalmanFilter object',
            pretty_str('dim_x', self.dim_x),
            pretty_str('dim_z', self.dim_z),
            pretty_str('dt', self.dt),
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('Q', self.Q),
            pretty_str('R', self.R),
            pretty_str('K', self.K),
            pretty_str('S', self.S),
            pretty_str('sigmas', self.sigmas),
            pretty_str('hx', self.hx),
            pretty_str('fx', self.fx)
            ])
