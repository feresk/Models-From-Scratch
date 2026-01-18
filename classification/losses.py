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
        if self.loss_fn == 'hinge' : return np.where(1-self.m>0, 1-self.m, 0).mean()
        elif self.loss_fn == 'squared_hinge' : return (np.where(1-self.m>0, 1-self.m, 0)**2).mean()
        elif self.loss_fn == 'log_loss' : return np.log(1+np.exp(-self.m)).mean()
        elif self.loss_fn == 'modified_hubor': return np.where(self.m>=-1, (1-self.m)**2, -4*self.m).mean()
        elif self.loss_fn == 'perceptron' : return np.where(-self.m>0, -self.m, 0).mean()

    def __call__(self, x, Y):
        y = 2*np.copy(Y)-1
        if len(y.shape) == 1 : y = y.reshape(-1,1)
        self.m = y*self.model(x)
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
    
    def _calc_grad(self, x, y, var='w'):
        if var == 'w':
            if self.loss_fn == 'hinge' : return -x.T @ np.where(1-self.m>0, y, 0)
            elif self.loss_fn == 'squared_hinge' : return -x.T @ np.where(1-self.m>0, 2*y*(1-self.m), 0)
            elif self.loss_fn == 'log_loss' : return -x.T @ (y/(np.exp(self.m) + 1))
            elif self.loss_fn == 'modified_hubor' : return -x.T @ np.where(self.m>=-1, 2*y*(1-self.m), 4*y)
            elif self.loss_fn == 'perceptron' : return -x.T @ np.where(-self.m>0, y, 0)
        elif var == 'b' : 
            if self.loss_fn == 'hinge' : return np.where(1-self.m>0, -y, 0).mean()
            elif self.loss_fn == 'squared_hinge' : return np.where(1-self.m>0, -2*y*(1-self.m), 0).mean()
            elif self.loss_fn == 'log_loss' : return -(y/(np.exp(self.m) + 1)).mean()
            elif self.loss_fn == 'modified_hubor' : return np.where(self.m>=-1, -2*y*(1-self.m), -4*y).mean()
            elif self.loss_fn == 'perceptron' : return np.where(-self.m>0, -y, 0).mean()

    def _backward(self, x, y):
        def _backward_w(x):
            w_grad = self._calc_grad(x, y, 'w')
            self.model.W -= self.model.eta * w_grad + self._regularization_grad()
        def _backward_b():
            b_grad = self._calc_grad(x, y, 'b')
            self.model.b -= self.model.eta * b_grad
        _backward_w(x)
        if self.model.fit_intercept : _backward_b()