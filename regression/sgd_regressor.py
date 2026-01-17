import numpy as np
import pandas as pd
from losses import Loss

class SGDRegressor():
    def __init__(self, loss='squared_error', penalty=None, alpha=1e-4, l1_ratio=.15, fit_intercept=True, max_iter=1000, tol=1e-3,
                 sample_size=.3, epsilon=0.1, random_state=None, learning_rate='optimal', eta0=.01, power_t=.25):
        self.W = None
        self.b = None
        self.params_set = False
        self.loss_fn = loss
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.sample_size = sample_size
        self.epsi = epsilon
        self.rd = random_state
        self.lr = learning_rate
        self.eta = eta0
        self.eta0 = eta0
        self.power_t = power_t

    def __call__(self, x): return x @ self.W + self.b
    
    def _random_generator(self):
        if self.rd is not None : self.rng = np.random.RandomState(self.rd)
        else : self.rng = np.random

    def _set_params(self,x):
        self._random_generator()
        self.params_set = True
        self.W = self.rng.normal(0, .5, size=(x.shape[1], 1))
        self.b = 0

    def _get_params(self): return (self.b, self.W)

    def _get_sample_batch(self, x):
        self.idx = self.rng.choice(range(x.shape[0]), int(x.shape[0]*self.sample_size), replace=False)

    def _update_eta(self, t, prev_loss, this_loss):
        if self.lr == 'constant' : None
        elif self.lr == 'optimal' : self.eta = 1 / (self.alpha * (t + 1/(self.alpha*self.eta0) ))
        elif self.lr == 'invscaling' : self.eta = self.eta0 / (t+1)**self.power_t
        elif self.lr == 'adaptive' :
            if t>0 and this_loss>prev_loss : self.eta /= 5

    def fit(self, X, y):
        x = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        if len(y.shape) == 1 : y = y.reshape(-1,1)
        if not self.params_set: self._set_params(x)
        self._get_sample_batch(x)
        self.loss = Loss(self)
        prev_loss = 0
        for t in range(self.max_iter):
            this_loss = self.loss(x[self.idx,:],y[self.idx,:])
            if t==0: self.loss._calc_grad(x[self.idx,:])
            self._update_eta(t, prev_loss, this_loss)
            if self.tol is not None and abs(this_loss-prev_loss)<self.tol : break
            prev_loss = this_loss
            self.loss._backward(x[self.idx,:])