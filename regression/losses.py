import numpy as np

class Loss():
    def __init__(self, model):
        self.model = model
        self.penalty = model.penalty
        self.alpha = model.alpha
        self.l1_ratio = model.l1_ratio
        self.loss_fn = model.loss_fn
        self.epsi = model.epsi

    def _calc_loss(self, x):
        if self.loss_fn == 'squared_error' : return (self.resid**2).mean()
        elif self.loss_fn == 'huber' :
            self.resid_idx_sm = self.resid <= self.epsi
            self.resid_idx_gt =  self.resid > self.epsi
            return ((self.resid[self.resid_idx_sm]**2).sum() + 
                        (self.epsi*np.abs(self.resid[self.resid_idx_gt])).sum())/len(x) - self.epsi**2
        elif self.loss_fn == 'epsilon_insensitive':
            return np.where(np.abs(self.resid)-self.epsi>0, np.abs(self.resid)-self.epsi, 0).mean() 
        elif self.loss_fn == 'squared_epsilon_insensitive' :
            return np.where(np.abs(self.resid)-self.epsi>0, self.resid**2-self.epsi, 0).mean() 

    def __call__(self, x ,y):
        if len(y.shape) == 1 : y = y.reshape(-1,1)
        self.resid = self.model(x)-y
        self.loss_out = self._calc_loss(x) + self._regularization_loss()
        return self.loss_out
    
    def _regularization_loss(self):
        if self.penalty == None:  return 0
        elif self.penalty == 'l1': return self.alpha * np.abs(self.model.W).sum()
        elif self.penalty == 'l2': return self.alpha * (self.model.W.T @ self.model.W)[0][0]
        elif self.penalty == 'elasticnet':
            return self.alpha * ((1-self.l1_ratio) * (self.model.W.T @ self.model.W)[0][0] + 
                                 self.l1_ratio * np.abs(self.model.W).sum())
        
    def _regularization_grad(self):
        if self.penalty == None : return 0
        elif self.penalty == 'l1': return self.alpha * np.sign(self.model.W)
        elif self.penalty == 'l2': return 2 * self.alpha * self.model.W
        elif self.penalty == 'elasticnet':
            return self.alpha * ((1-self.l1_ratio)*np.sign(self.model.W) +
                                  self.l1_ratio*2*self.model.W)
        return None
    
    def _calc_grad(self, x, var='w'):
        if var == 'w':
            if self.loss_fn == 'squared_error' : return 2/len(x) * (x.T @ self.resid)
            elif self.loss_fn == 'huber' :
                w_grad_1 =  x.T[:,self.resid_idx_sm.ravel()] @ self.resid[self.resid_idx_sm.ravel(),:]
                w_grad_2 =  self.epsi * x.T[:,self.resid_idx_gt.ravel()] @ np.sign(self.resid[self.resid_idx_gt.ravel(),:])
                return (w_grad_1 + w_grad_2)/len(x)
            elif self.loss_fn == 'epsilon_insensitive' :
                return x.T @ np.where(np.abs(self.resid)-self.epsi>0, np.sign(self.resid), 0)
            elif self.loss_fn == 'squared_epsilon_insensitive' :
                return x.T @ np.where(np.abs(self.resid)-self.epsi>0, self.resid, 0)
        elif var == 'b':
            if self.loss_fn == 'squared_error' : return 2 * self.resid.mean()
            elif self.loss_fn == 'huber' :
                b_grad_1 = self.resid[self.resid_idx_sm.ravel(),:].mean()
                b_grad_2 = self.epsi * np.sign(self.resid)[self.resid_idx_sm.ravel(),:].mean()
                return b_grad_1 + b_grad_2
            elif self.loss_fn == 'epsilon_insensitive' :
                return np.where(np.abs(self.resid)-self.epsi>0, np.sign(self.resid), 0).mean()
            elif self.loss_fn == 'squared_epsilon_insensitive' :
                return np.where(np.abs(self.resid)-self.epsi>0, self.resid, 0).mean()

    def _backward(self, x):
        def _backward_w(x):
            w_grad = self._calc_grad(x, 'w')
            self.model.W -= self.model.eta * w_grad + self._regularization_grad()
        def _backward_b():
            b_grad = self._calc_grad(x, 'b')
            self.model.b -= self.model.eta * b_grad
        _backward_w(x)
        if self.model.fit_intercept : _backward_b()