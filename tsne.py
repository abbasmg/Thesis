# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 07:57:36 2020

@author: abbme
"""

import smilefrown

a = smilefrown.l_smile_frown(4)

a.shape

import theano.tensor as T
import theano
import numpy as np

epsilon = 1e-16
floath = np.float32


def sqeuclidean_var(X):
    N = X.shape[0]
    ss = (X ** 2).sum(axis=1)
    return ss.reshape((N, 1)) + ss.reshape((1, N)) - 2*X.dot(X.T)


def p_Xp_given_X_var(X, sigma, metric):
    N = X.shape[0]

    if metric == 'euclidean':
        sqdistance = sqeuclidean_var(X)
    elif metric == 'precomputed':
        sqdistance = X**2
    else:
        raise Exception('Invalid metric')

    esqdistance = T.exp(-sqdistance / ((2 * (sigma**2)).reshape((N, 1))))
    esqdistance_zd = T.fill_diagonal(esqdistance, 0)

    row_sum = T.sum(esqdistance_zd, axis=1).reshape((N, 1))

    return esqdistance_zd/row_sum  # Possibly dangerous


def p_Xp_X_var(p_Xp_given_X):
    return (p_Xp_given_X + p_Xp_given_X.T) / (2 * p_Xp_given_X.shape[0])


def p_Yp_Y_var(Y):
    sqdistance = sqeuclidean_var(Y)
    one_over = T.fill_diagonal(1/(sqdistance + 1), 0)
    return one_over/one_over.sum()  # Possibly dangerous

    
def cost_var(X, Y, sigma, metric):
    p_Xp_given_X = p_Xp_given_X_var(X, sigma, metric)
    PX = p_Xp_X_var(p_Xp_given_X)
    PY = p_Yp_Y_var(Y)
    
    PXc = T.maximum(PX, epsilon)
    PYc = T.maximum(PY, epsilon)
    return T.sum(PX * T.log(PXc / PYc))  # Possibly dangerous (clipped)


def find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters,
               metric, verbose=0):
    """Binary search on sigma for a given perplexity."""
    X = T.fmatrix('X')
    sigma = T.fvector('sigma')

    target = np.log(perplexity)

    P = T.maximum(p_Xp_given_X_var(X, sigma, metric), epsilon)

    entropy = -T.sum(P*T.log(P), axis=1)

    # Setting update for binary search interval
    sigmin_shared = theano.shared(np.full(N, np.sqrt(epsilon), dtype=floath))
    sigmax_shared = theano.shared(np.full(N, np.inf, dtype=floath))

    sigmin = T.fvector('sigmin')
    sigmax = T.fvector('sigmax')

    upmin = T.switch(T.lt(entropy, target), sigma, sigmin)
    upmax = T.switch(T.gt(entropy, target), sigma, sigmax)

    givens = {X: X_shared, sigma: sigma_shared, sigmin: sigmin_shared,
              sigmax: sigmax_shared}
    updates = [(sigmin_shared, upmin), (sigmax_shared, upmax)]

    update_intervals = theano.function([], entropy, givens=givens,
                                       updates=updates)

    # Setting update for sigma according to search interval
    upsigma = T.switch(T.isinf(sigmax), sigma*2, (sigmin + sigmax)/2.)

    givens = {sigma: sigma_shared, sigmin: sigmin_shared,
              sigmax: sigmax_shared}
    updates = [(sigma_shared, upsigma)]

    update_sigma = theano.function([], sigma, givens=givens, updates=updates)

    for i in range(sigma_iters):
        e = update_intervals()
        update_sigma()
        if verbose:
            print('Iteration: {0}.'.format(i+1))
            print('Perplexities in [{0:.4f}, {1:.4f}].'.format(np.exp(e.min()),
                  np.exp(e.max())))

    if np.any(np.isnan(np.exp(e))):
        raise Exception('Invalid sigmas. The perplexity is probably too low.')
        
        
import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import check_random_state




def find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
           initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
           final_momentum, momentum_switch, metric, verbose=0):
    """Optimize cost wrt Y"""
    # Optimization hyperparameters
    initial_lr = np.array(initial_lr, dtype=floath)
    final_lr = np.array(final_lr, dtype=floath)
    initial_momentum = np.array(initial_momentum, dtype=floath)
    final_momentum = np.array(final_momentum, dtype=floath)

    lr = T.fscalar('lr')
    lr_shared = theano.shared(initial_lr)

    momentum = T.fscalar('momentum')
    momentum_shared = theano.shared(initial_momentum)

    # Y velocities
    Yv = T.fmatrix('Yv')
    Yv_shared = theano.shared(np.zeros((N, output_dims), dtype=floath))

    # Cost
    X = T.fmatrix('X')
    sigma = T.fvector('sigma')
    Y = T.fmatrix('Y')

    cost = cost_var(X, Y, sigma, metric)

    # Setting update for Y velocities
    grad_Y = T.grad(cost, Y)

    updates = [(Yv_shared, momentum*Yv - lr*grad_Y)]
    givens = {X: X_shared, sigma: sigma_shared, Y: Y_shared, Yv: Yv_shared,
              lr: lr_shared, momentum: momentum_shared}

    update_Yv = theano.function([], cost, givens=givens, updates=updates)

    # Setting update for Y
    givens = {Y: Y_shared, Yv: Yv_shared}
    updates = [(Y_shared, Y + Yv)]

    update_Y = theano.function([], [], givens=givens, updates=updates)

    # Momentum-based gradient descent
    for epoch in range(n_epochs):
        if epoch == lr_switch:
            lr_shared.set_value(final_lr)
        if epoch == momentum_switch:
            momentum_shared.set_value(final_momentum)

        c = update_Yv()
        update_Y()
        if verbose:
            print('Epoch: {0}. Cost: {1:.6f}.'.format(epoch + 1, float(c)))

    return np.array(Y_shared.get_value())


def tsne(X, perplexity=30, Y=None, output_dims=2, n_epochs=1000,
         initial_lr=2400, final_lr=200, lr_switch=250, init_stdev=1e-4,
         sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
         momentum_switch=250, metric='euclidean', random_state=None,
         verbose=1):
    """Compute projection from a matrix of observations (or distances) using 
    t-SNE.
    
    Parameters
    ----------
    X : array-like, shape (n_observations, n_features), \
            or (n_observations, n_observations) if `metric` == 'precomputed'.
        Matrix containing the observations (one per row). If `metric` is 
        'precomputed', pairwise dissimilarity (distance) matrix.
    
    perplexity : float, optional (default = 30)
        Target perplexity for binary search for sigmas.
        
    Y : array-like, shape (n_observations, output_dims), optional \
            (default = None)
        Matrix containing the starting position for each point.
    
    output_dims : int, optional (default = 2)
        Target dimension.
        
    n_epochs : int, optional (default = 1000)
        Number of gradient descent iterations.
        
    initial_lr : float, optional (default = 2400)
        The initial learning rate for gradient descent.
        
    final_lr : float, optional (default = 200)
        The final learning rate for gradient descent.
        
    lr_switch : int, optional (default = 250)
        Iteration in which the learning rate changes from initial to final.
        This option effectively subsumes early exaggeration.
        
    init_stdev : float, optional (default = 1e-4)
        Standard deviation for a Gaussian distribution with zero mean from
        which the initial coordinates are sampled.
        
    sigma_iters : int, optional (default = 50)
        Number of binary search iterations for target perplexity.
        
    initial_momentum : float, optional (default = 0.5)
        The initial momentum for gradient descent.
        
    final_momentum : float, optional (default = 0.8)
        The final momentum for gradient descent.
        
    momentum_switch : int, optional (default = 250)
        Iteration in which the momentum changes from initial to final.
        
    metric : 'euclidean' or 'precomputed', optional (default = 'euclidean')
        Indicates whether `X` is composed of observations ('euclidean') 
        or distances ('precomputed').
    
    random_state : int or np.RandomState, optional (default = None)
        Integer seed or np.RandomState object used to initialize the
        position of each point. Defaults to a random seed.
    verbose : bool (default = 1)
        Indicates whether progress information should be sent to standard 
        output.
        
    Returns
    -------
    Y : array-like, shape (n_observations, output_dims)
        Matrix representing the projection. Each row (point) corresponds to a
        row (observation or distance to other observations) in the input matrix.
    """
    random_state = check_random_state(random_state)

    N = X.shape[0]

    X_shared = theano.shared(np.asarray(X, dtype=floath))
    sigma_shared = theano.shared(np.ones(N, dtype=floath))

    if Y is None:
        Y = random_state.normal(0, init_stdev, size=(N, output_dims))
    Y_shared = theano.shared(np.asarray(Y, dtype=floath))

    find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters, metric,
               verbose)

    Y = find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
               initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
               final_momentum, momentum_switch, metric, verbose)

    return Y




def movement_penalty(Ys, N):
    penalties = []
    for t in range(len(Ys) - 1):
        penalties.append(T.sum((Ys[t] - Ys[t + 1])**2))

    return T.sum(penalties)/(2*N)


def find_Ys(Xs_shared, Ys_shared, sigmas_shared, N, steps, output_dims,
            n_epochs, initial_lr, final_lr, lr_switch, init_stdev,
            initial_momentum, final_momentum, momentum_switch, lmbda, metric,
            verbose=0):
    """Optimize cost wrt Ys[t], simultaneously for all t"""
    # Optimization hyperparameters
    initial_lr = np.array(initial_lr, dtype=floath)
    final_lr = np.array(final_lr, dtype=floath)
    initial_momentum = np.array(initial_momentum, dtype=floath)
    final_momentum = np.array(final_momentum, dtype=floath)

    lr = T.fscalar('lr')
    lr_shared = theano.shared(initial_lr)

    momentum = T.fscalar('momentum')
    momentum_shared = theano.shared(initial_momentum)

    # Penalty hyperparameter
    lmbda_var = T.fscalar('lmbda')
    lmbda_shared = theano.shared(np.array(lmbda, dtype=floath))

    # Yv velocities
    Yvs_shared = []
    zero_velocities = np.zeros((N, output_dims), dtype=floath)
    for t in range(steps):
        Yvs_shared.append(theano.shared(np.array(zero_velocities)))

    # Cost
    Xvars = T.fmatrices(steps)
    Yvars = T.fmatrices(steps)
    Yv_vars = T.fmatrices(steps)
    sigmas_vars = T.fvectors(steps)

    c_vars = []
    for t in range(steps):
        c_vars.append(cost_var(Xvars[t], Yvars[t], sigmas_vars[t], metric))

    cost = T.sum(c_vars) + lmbda_var*movement_penalty(Yvars, N)

    # Setting update for Ys velocities
    grad_Y = T.grad(cost, Yvars)

    givens = {lr: lr_shared, momentum: momentum_shared,
              lmbda_var: lmbda_shared}
    updates = []
    for t in range(steps):
        updates.append((Yvs_shared[t], momentum*Yv_vars[t] - lr*grad_Y[t]))

        givens[Xvars[t]] = Xs_shared[t]
        givens[Yvars[t]] = Ys_shared[t]
        givens[Yv_vars[t]] = Yvs_shared[t]
        givens[sigmas_vars[t]] = sigmas_shared[t]

    update_Yvs = theano.function([], cost, givens=givens, updates=updates)

    # Setting update for Ys positions
    updates = []
    givens = dict()
    for t in range(steps):
        updates.append((Ys_shared[t], Yvars[t] + Yv_vars[t]))
        givens[Yvars[t]] = Ys_shared[t]
        givens[Yv_vars[t]] = Yvs_shared[t]

    update_Ys = theano.function([], [], givens=givens, updates=updates)

    # Momentum-based gradient descent
    for epoch in range(n_epochs):
        if epoch == lr_switch:
            lr_shared.set_value(final_lr)
        if epoch == momentum_switch:
            momentum_shared.set_value(final_momentum)

        c = update_Yvs()
        update_Ys()
        if verbose:
            print('Epoch: {0}. Cost: {1:.6f}.'.format(epoch + 1, float(c)))

    Ys = []
    for t in range(steps):
        Ys.append(np.array(Ys_shared[t].get_value(), dtype=floath))

    return Ys


def dynamic_tsne(Xs, perplexity=30, Ys=None, output_dims=2, n_epochs=1000,
                 initial_lr=2400, final_lr=200, lr_switch=250, init_stdev=1e-4,
                 sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
                 momentum_switch=250, lmbda=0.0, metric='euclidean',
                 random_state=None, verbose=1):
    """Compute sequence of projections from a sequence of matrices of
    observations (or distances) using dynamic t-SNE.
    
    Parameters
    ----------
    Xs : list of array-likes, each with shape (n_observations, n_features), \
            or (n_observations, n_observations) if `metric` == 'precomputed'.
        List of matrices containing the observations (one per row). If `metric` 
        is 'precomputed', list of pairwise dissimilarity (distance) matrices. 
        Each row in `Xs[t + 1]` should correspond to the same row in `Xs[t]`, 
        for every time step t > 1.
    
    perplexity : float, optional (default = 30)
        Target perplexity for binary search for sigmas.
        
    Ys : list of array-likes, each with shape (n_observations, output_dims), \
            optional (default = None)
        List of matrices containing the starting positions for each point at
        each time step.
    
    output_dims : int, optional (default = 2)
        Target dimension.
        
    n_epochs : int, optional (default = 1000)
        Number of gradient descent iterations.
        
    initial_lr : float, optional (default = 2400)
        The initial learning rate for gradient descent.
        
    final_lr : float, optional (default = 200)
        The final learning rate for gradient descent.
        
    lr_switch : int, optional (default = 250)
        Iteration in which the learning rate changes from initial to final.
        This option effectively subsumes early exaggeration.
        
    init_stdev : float, optional (default = 1e-4)
        Standard deviation for a Gaussian distribution with zero mean from
        which the initial coordinates are sampled.
        
    sigma_iters : int, optional (default = 50)
        Number of binary search iterations for target perplexity.
        
    initial_momentum : float, optional (default = 0.5)
        The initial momentum for gradient descent.
        
    final_momentum : float, optional (default = 0.8)
        The final momentum for gradient descent.
        
    momentum_switch : int, optional (default = 250)
        Iteration in which the momentum changes from initial to final.
        
    lmbda : float, optional (default = 0.0)
        Movement penalty hyperparameter. Controls how much each point is
        penalized for moving across time steps.
        
    metric : 'euclidean' or 'precomputed', optional (default = 'euclidean')
        Indicates whether `X[t]` is composed of observations ('euclidean') 
        or distances ('precomputed'), for all t.
    
    random_state : int or np.RandomState, optional (default = None)
        Integer seed or np.RandomState object used to initialize the
        position of each point. Defaults to a random seed.
    verbose : bool (default = 1)
        Indicates whether progress information should be sent to standard 
        output.
        
    Returns
    -------
    Ys : list of array-likes, each with shape (n_observations, output_dims)
        List of matrices representing the sequence of projections. 
        Each row (point) in `Ys[t]` corresponds to a row (observation or 
        distance to other observations) in the input matrix `Xs[t]`, for all t.
    """
    random_state = check_random_state(random_state)

    steps = len(Xs)
    N = Xs[0].shape[0]

    if Ys is None:
        Y = random_state.normal(0, init_stdev, size=(N, output_dims))
        Ys = [Y]*steps

    for t in range(steps):
        if Xs[t].shape[0] != N or Ys[t].shape[0] != N:
            raise Exception('Invalid datasets.')

        Xs[t] = np.array(Xs[t], dtype=floath)

    Xs_shared, Ys_shared, sigmas_shared = [], [], []
    for t in range(steps):
        X_shared = theano.shared(Xs[t])
        sigma_shared = theano.shared(np.ones(N, dtype=floath))

        find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters,
                   metric=metric, verbose=verbose)

        Xs_shared.append(X_shared)
        Ys_shared.append(theano.shared(np.array(Ys[t], dtype=floath)))
        sigmas_shared.append(sigma_shared)

    Ys = find_Ys(Xs_shared, Ys_shared, sigmas_shared, N, steps, output_dims,
                 n_epochs, initial_lr, final_lr, lr_switch, init_stdev,
                 initial_momentum, final_momentum, momentum_switch, lmbda,
                 metric, verbose)

    return Ys

tpX = a[:,:-1]
tpY = a[:,-1]

from sklearn.utils import shuffle

X, y = shuffle(tpX, tpY)


Y = tsne(X, perplexity=70, n_epochs=1000, sigma_iters=50,
             random_state=12345, verbose=1)
import matplotlib as plt
import pylab
hsv_colors = [(0.56823266219239377, 0.82777777777777772, 0.70588235294117652),
              (0.078146611341632088, 0.94509803921568625, 1.0),
              (0.33333333333333331, 0.72499999999999998, 0.62745098039215685),
              (0.99904761904761907, 0.81775700934579443, 0.83921568627450982),
              (0.75387596899224807, 0.45502645502645506, 0.74117647058823533),
              (0.028205128205128216, 0.4642857142857143, 0.5490196078431373),
              (0.8842592592592593, 0.47577092511013214, 0.8901960784313725),
              (0.0, 0.0, 0.49803921568627452),
              (0.16774193548387095, 0.82010582010582012, 0.74117647058823533),
              (0.51539855072463769, 0.88888888888888884, 0.81176470588235294)]

rgb_colors = plt.colors.hsv_to_rgb(np.array(hsv_colors).reshape(10, 1, 3))
colors = plt.colors.ListedColormap(rgb_colors.reshape(10, 3))



pylab.scatter(Y[:, 0], Y[:, 1], s=30, c=y, cmap=colors, linewidth=0)
pylab.show()
m,n = X.shape
Xs = list(X.reshape((n,m,1)))
Ys = dynamic_tsne(Xs, perplexity=20, lmbda=0.1, verbose=1, sigma_iters=50,
                      random_state=12345)

def create_blobs(classes=10, dims=100, class_size=100, variance=0.1, steps=4,
                 advection_ratio=0.5, random_state=None):
    random_state = check_random_state(random_state)
    X = []

    indices = random_state.permutation(dims)[0:classes]
    means = []
    for c in range(classes):
        mean = np.zeros(dims)
        mean[indices[c]] = 1.0
        means.append(mean)

        X.append(random_state.multivariate_normal(mean, np.eye(dims)*variance,
                                                  class_size))
    X = np.concatenate(X)
    y = np.concatenate([[i]*class_size for i in range(classes)])

    Xs = [np.array(X)]
    for step in range(steps - 1):
        Xnext = np.array(Xs[step])
        for c in range(classes):
            stard, end = class_size*c, class_size*(c + 1)
            Xnext[stard: end] += advection_ratio*(means[c] - Xnext[stard: end])

        Xs.append(Xnext)

    return Xs, y



seed = 0

Xs, y = create_blobs(class_size=200, advection_ratio=0.1, steps=10,
                          random_state=seed)



Ys = dynamic_tsne(Xs, perplexity=70, lmbda=0.1, verbose=1, sigma_iters=50,
                      random_state=seed)



for Y in Ys:
    pylab.scatter(Y[:, 0], Y[:, 1], s=30, c=y, cmap=colors, linewidth=0)
    pylab.show()

    