import numpy as np


class HamiltonianMC:
    def __init__(self, log_prob_fn, log_prob_grad_fn, n_steps = None, step_size = 0.01):
                
        self.log_prob_fn = log_prob_fn
        self.log_prob_grad_fn = log_prob_grad_fn
        
        self.step_size = step_size
        self.n_steps = n_steps

        
    def _n_steps(self):
        if self.n_steps == None:
            return np.random.randint(11, 30, 1)[0]
        return self.n_steps
    
    def _kinetic_energy(self, momentum):
        return 0.5 * np.sum(momentum**2)
    
    def _leapfrog_step(self, position, momentum, direction=1):
        
        momentum = momentum + 0.5 * self.step_size * self.log_prob_grad_fn(position)
        position = position + self.step_size * momentum * direction
        momentum =  momentum + 0.5 * self.step_size * self.log_prob_grad_fn(position)
        return position, momentum  

    def next_proposal(self, position, momentum):
        
        for _ in range(self._n_steps()):
            position, momentum = self._leapfrog_step(position, momentum)
            
        return position, momentum
            
    def sample(self, current_position, n_samples, burn_in=1000):
        
        dim = len(current_position)        
        samples = np.zeros((n_samples,dim))
        
        current_log_prob = self.log_prob_fn(current_position)
        
        
        for i in range(n_samples):
            
            momentum = np.random.normal(0, 1, dim)
            current_hamiltonian = -current_log_prob + self._kinetic_energy(momentum)
            
            next_position, next_momentum = self.next_proposal(current_position, momentum)
            
            next_log_prob = self.log_prob_fn(next_position)
            next_hamiltonian = -next_log_prob + self._kinetic_energy(next_momentum)
            
            log_accaptance_ratio = current_hamiltonian-next_hamiltonian
            
            if log_accaptance_ratio > np.log(np.random.random()):
                
                current_position = next_position
                current_log_prob = next_log_prob
                
               
            samples[i] = current_position
            
        return samples[burn_in:]
            
            
            