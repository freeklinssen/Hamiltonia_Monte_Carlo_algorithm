import numpy as np
import scipy 

class HamiltonianMC_no_uturn:
    def __init__(self, log_prob_fn, log_prob_grad_fn):
                
        self.log_prob_fn = log_prob_fn
        self.log_prob_grad_fn = log_prob_grad_fn
        
        self.step_size = 0.1
        self.step_size_ = 0.1
        self.log_step_size_ = 0
        
        #step_size adjustment
        self.mu = 0
        self.H = 0
        self.gamma = 0.05
        self.t_0 = 10
        self.k = 0.75


    
    def _find_reasonable_step_size(self, theta):
        self.step_size = 0.1
        self.step_size_ = 0.1
        r = np.random.normal(0, 1, len(theta))
       
        theta_ , r_ = self._leapfrog_step(theta, r)
        P_theta_r = self.log_prob_fn(theta)- self._kinetic_energy(r)
        P_theta__r_ = self.log_prob_fn(theta_)- self._kinetic_energy(r_)

        ratio = np.exp(P_theta__r_ - P_theta_r)
        a = 2 * int(ratio > 0.5) -1 # +1 if ratio > 0.5, -1 otherwise
        
        while ratio**a > 2**(-a):
            self.step_size = 2**a * self.step_size
            theta_, r_ = self._leapfrog_step(theta, r)
            P_theta__r_ = self.log_prob_fn(theta_) - self._kinetic_energy(r_)
            ratio = np.exp(P_theta__r_ - P_theta_r)
        
        self.mu = np.log(10*self.step_size)
        
        
    def _kinetic_energy(self, momentum):
        return 0.5 * np.sum(momentum ** 2)
    
    
    def _leapfrog_step(self, position, momentum, direction=1):
        
        momentum = momentum + 0.5 * self.step_size * self.log_prob_grad_fn(position) * direction
        position = position + self.step_size * momentum * direction
        #position = np.clip(position + self.step_size * momentum * direction, 0, None)
        momentum =  momentum + 0.5 * self.step_size * self.log_prob_grad_fn(position) * direction
        return position, momentum
    
    
    def _check_uturn(self, leftmost_pos, rightmost_pos, leftmost_mom, rightmost_mom):
        # check if points get closer to each other
        diff = rightmost_pos - leftmost_pos
        return np.dot(diff, leftmost_mom) >= 0 * np.dot(diff, rightmost_mom) >= 0 
            

    def _build_tree(self, theta, r, u, direction, depth):
        
        if depth == 0:
            next_theta, next_r = self._leapfrog_step(theta, r, direction)

            log_prob_next_theta=self.log_prob_fn(next_theta) 
            log_prob_next_theta = np.where(np.isnan(log_prob_next_theta), -10e40, log_prob_next_theta)
            
            n_ = 1 if u <= np.exp(log_prob_next_theta - self._kinetic_energy(next_r)) else 0
            s_ = 1 if log_prob_next_theta - self._kinetic_energy(next_r) - np.log(u) > -1000 else 0 # the error is large in too large in this case

            return next_theta, next_r, next_theta, next_r, next_theta, n_, s_, min(1, np.exp(log_prob_next_theta - self._kinetic_energy(next_r) - self.log_prob_fn(theta) + self._kinetic_energy(r))), 1
        
        else:
            theta_neg, r_neg, theta_pos, r_pos, theta_, n_, s_, alpha_, n_alpha   = self._build_tree(theta, r, u, direction, depth-1)
            
            if s_ == 1: 
                if direction == -1:
                    theta_neg, r_neg, _, _, theta__, n__, s__, alpha__, n__alpha = self._build_tree(theta_neg, r_neg, u, -1, depth-1)
                else:
                    _, _, theta_pos, r_pos, theta__, n__, s__, alpha__, n__alpha = self._build_tree(theta_pos, r_pos, u, 1, depth-1)

                if n_==0 and n__ ==0:
                    theta_ = theta__ if np.random.choice([1, 0], p=[0.5, 0.5]) else theta_
                else:
                    theta_ = theta__ if np.random.choice([1, 0], p=[n__/(n_+n__), 1-(n__/(n_+n__))]) else theta_    
                n_ =  n_ + n__
                s_ = s__ * self._check_uturn(theta_neg, theta_pos, r_neg, r_pos)
                
                alpha_ = alpha_ + alpha__
                n_alpha = n_alpha + n__alpha
                
            return theta_neg, r_neg, theta_pos, r_pos, theta_, n_, s_, alpha_, n_alpha 
                    
                    
                        
    def sample(self, theta_m, n_samples, burn_in, step_size =None, update_step_size = True):      
        samples = np.zeros((n_samples,len(theta_m)))
        
        if step_size == None:
            self._find_reasonable_step_size(theta_m)
        else:
            self.step_size=step_size

        
        for i in range(n_samples):
            r_neg = r_pos = np.random.normal(0, 1, len(theta_m))
            theta_neg = theta_pos = theta_m
            u = np.random.uniform(0, np.exp(self.log_prob_fn(theta_m) - self._kinetic_energy(r_pos)))

            depth = 0
            valid = 1
            n = 1
            
            while valid == 1 and depth<5:
                direction = np.random.choice([1, -1])
                if direction == -1:
                    theta_neg, r_neg, _, _, theta_, n_, s_ , alpha, Nalpha = self._build_tree(theta_neg, r_neg, u, -1, depth)
                else:
                    _, _, theta_pos, r_pos, theta_, n_, s_ , alpha, Nalpha = self._build_tree(theta_pos, r_pos, u, 1, depth)  
                
                if s_ == 1:
                    theta_m = theta_ if np.random.choice([1, 0], p=[min(1, n_/n), 1-min(1, n_/n)]) else theta_m
                n = n + n_ 
                valid = s_ * self._check_uturn(theta_neg, theta_pos, r_neg, r_pos)
                depth += 1
                
            samples[i] = theta_m
            
            # step size adjustment
            if update_step_size == True:
                if i < 0.2*n_samples:
                    self.H = (1-1/((i+1)+self.t_0)) * self.H + (1/((i+1)+self.t_0))*(0.9 - alpha/Nalpha)  # 0.90 is target mean accaptance probability

                    log_step_size = self.mu - (np.sqrt(i+1)/self.gamma)*self.H
                    self.step_size = np.exp(log_step_size)
                    
                    self.log_step_size_  = (i+1)**(-self.k) * log_step_size + (1-(i+1)**(-self.k)) * self.log_step_size_ 
                    self.step_size_ = np.exp(self.log_step_size_ )
                    
                if i >= 0.2*n_samples:
                    self.step_size = self.step_size_
            
        return samples[burn_in:]  