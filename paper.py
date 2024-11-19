import numpy as np
import scipy 

class HamiltonianMC:
    def __init__(self, log_prob_fn, log_prob_grad_fn, step_size = 0.1, n_steps = 20):
                
        self.log_prob_fn = log_prob_fn
        self.log_prob_grad_fn = log_prob_grad_fn
        
        self.step_size = step_size
        self.n_steps = n_steps

        
    def _n_steps(self):
        if self.n_steps == None:
            return np.random.randint(11, 30, 1)[0]
        return self.n_steps
    
    def _kinetic_energy(self, momentum):
        return 0.5 * np.sum(momentum ** 2)
    
    def _leapfrog_step(self, position, momentum, direction=1):
        
        momentum = momentum + 0.5 * self.step_size * self.log_prob_grad_fn(position)
        position = position + self.step_size * momentum * direction
        momentum =  momentum + 0.5 * self.step_size * self.log_prob_grad_fn(position)
        return position, momentum
    
    def _check_uturn(leftmost_pos, rightmost_pos, leftmost_mom, rightmost_mom):
        # Simple version: check if points getting closer
        diff = rightmost_pos - leftmost_pos
        return np.dot(diff, leftmost_mom) >= 0 * np.dot(diff, rightmost_mom) >= 0 
            

    def _build_tree(self, theta, r, u, direction, depth):
        
        if depth == 0:
            next_theta, next_r = self._leapfrog_step(theta, r, direction)
            n_ = 1 if u <= np.exp(self.log_prob_fn(next_theta) - self._kinetic_energy(next_r)) else 0
            s_ = 1 if u < np.exp(1000 + self.log_prob_fn(next_theta) - self._kinetic_energy(next_r)) else 0 # the error is large in too large in this case

            return next_theta, next_r, next_theta, next_r, next_theta, n_, s_
        
        else:
            theta_neg, r_neg, theta_pos, r_pos, theta_, n_, s_ = self._build_tree(theta, r, u, direction, depth-1)
            
            if s_ == 1: 
                if direction == -1:
                    theta_neg, r_neg, _, _, theta__, n__, s__ = self._build_tree(theta_neg, r_neg, u, -1, depth-1)
                else:
                    _, _, theta_pos, r_pos, theta__, n__, s__ = self._build_tree(theta_pos, r_pos, u, 1, depth-1)
            
                theta_ = theta__ if np.random.choice([1, 0], p=[n__/(n_+n__), 1-(n__/(n_+n__))]) else theta_    
                n_ =  n_ + n__
                s_ = s__ * self._check_uturn(theta_neg, theta_pos, r_neg, r_pos)
                
                return theta_neg, r_neg, theta_pos, r_pos, theta_, n_, s_
                
            else: 
                return theta_neg, r_neg, theta_pos, r_pos, theta_, n_, s_
                    
                    
            
    '''
    def next_proposal(self, position, momentum):
        
        for _ in range(self.n_steps):
            position, momentum = self._leapfrog_step(position, momentum)
            
        return position, momentum
    ''' 
        
    def sample(self, theta_m, n_samples, burn_in = 1000):
      
        samples = np.zeros((n_samples,len(theta_m)))
        
        for i in n_samples:
            r_neg = r_pos = np.random.normal(0, 1, len(theta_m))
            theta_neg = theta_pos = theta_m
            u = np.random.uniform(0, np.exp(self.log_prob_fn(theta_m) - self._kinetic_energy(r_neg)))
            depth = 0
            valid = 1
            n = 1
            
            while valid == 1:
                direction = np.random.choice([1, -1])
                if direction == -1:
                    theta_neg, r_neg, _, _, theta_, n_, s_  = self._build_tree(theta_neg, r_neg, u, -1, depth)
                else:
                    _, _, theta_pos, r_pos, theta_, n_, s_  = self._build_tree(theta_pos, r_pos, u, 1, depth)  
                
                if s_ == 1:
                    theta_m = theta_ if np.random.choice([1, 0], p=[min(1, n_/n), 1-min(1, n_/n)]) else theta_m
                n = n+n_ 
                valid = s_ * self._check_uturn(theta_neg, theta_pos, r_neg, r_pos)
                debt += 1
                
            samples[i] = theta_m
            
        return samples[burn_in:]
            
            
            
            