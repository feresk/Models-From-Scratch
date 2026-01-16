import numpy as np
import pandas as pd

class SGDRegression():
    def __init__(self, loss='squared_error', penalty=None, alpha=1e-4, fit_intercept=True, max_iter=1000, tol=1e-3,
                 sample_size=.3, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=.01, power_t=.25):
        self.weights = None
        self.intercept = None
        self.params_set = False
        self.loss_fn = loss
        self.reg = penalty
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.epochs = max_iter
        self.tol = tol
        self.sample_size = sample_size 
        self.epsi = epsilon
        self.rd = random_state
        self.lr = learning_rate
        self.eta, self.eta0  = eta0, eta0
        self.power_t = power_t

    def _set_params(self, x, y):
        self.batch_size, self.n_features = x.shape
        if self.rd is not None: 
            self.rng = np.random.RandomState(self.rd)
            self.weights = self.rng.normal(0, .5, size=(self.n_features, 1))
        else : 
            self.weights = np.random.normal(0, .5, size=(self.n_features, 1))
        self.intercept = 0 if self.fit_intercept else None
        self.params_set = True

    def __call__(self, x):
        return np.dot(self.weights.T, x.T) + (self.intercept if self.fit_intercept else 0)
    
    def loss(self, x, y):
        y_pred = self(x)
        self.resid = y_pred-y 
        if self.loss_fn == 'squared_error':
            self.mse = (self.resid**2).mean() 
        elif self.loss_fn == 'huber':
            self.resid_idx_sm = self.resid <= self.epsi
            self.resid_idx_gt =  self.resid > self.epsi
            self.mse = ((self.resid[self.resid_idx_sm]**2).sum() + 
                        (self.epsi*np.abs(self.resid[self.resid_idx_gt])).sum())/len(self.idx) - self.epsi**2
        if self.reg == 'l1' : self.mse += .5 * self.alpha * np.abs(self.weights).sum()
        elif self.reg == 'l2' : self.mse += .5 * self.alpha * np.dot(self.weights.T, self.weights)[0][0]
        return self.mse
    
    def backward(self, x):
        if self.loss_fn == 'squared_error': 
         weights_grad = (self.resid*x.T).mean(axis=1).reshape(-1,1)
        elif self.loss_fn == 'huber':
            weights_grad_first = (self.resid*x.T)[:,*self.resid_idx_sm].sum(axis=1)
            weights_grad_second = (self.epsi*np.sign(self.resid)*x.T)[:,*self.resid_idx_gt].sum(axis=1)
            weights_grad = ((weights_grad_first + weights_grad_second)/len(self.idx)).reshape(-1,1)
        if self.reg == 'l1' : weights_grad += self.alpha * np.sign(self.weights)
        elif self.reg == 'l2': weights_grad += self.alpha * self.weights 
        self.weights -= self.eta * weights_grad
        if self.fit_intercept : 
            bias_grad = self.resid.mean()
            self.intercept -= self.eta * bias_grad

    def fit(self, X, y):
        x = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        if not self.params_set : self._set_params(x,y)
        prev_loss = 0
        t0 = 1 / (self.alpha*self.eta)
        for t in range(self.epochs):
            if self.rd is not None: 
                self.idx = self.rng.choice(range(self.batch_size), int(self.batch_size*self.sample_size), replace=False)
            else : 
                self.idx = np.random.choice(range(self.batch_size), int(self.batch_size*self.sample_size), replace=False)
            this_loss = self.loss(x[self.idx,:], y[self.idx])
            if self.lr == 'optimal': self.eta = 1 / (self.alpha * (t+t0))
            elif self.lr == 'invscaling': self.eta = self.eta0 / np.pow(t+1, self.power_t)
            elif self.lr == 'adaptive' : 
                if t==0 : self.eta = self.eta0
                elif this_loss>prev_loss : self.eta /= 5 
            if self.tol is not None and abs(this_loss-prev_loss) < self.tol: break
            else : prev_loss = this_loss
            self.backward(x[self.idx,:])