import numpy as np


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
        return np.dot(diff, leftmost_mom) < 0 or np.dot(diff, rightmost_mom)< 0 
    
    
    def _build_tree(self, position, momentum, start_h, direction, depth):
        
        # direction: +1 for forward, -1 for backward in time
        if depth == 0:
            new_pos, new_mom = self._leapfrog_step(position, momentum, direction)
        
            new_h = -self.log_prob_fn(new_pos) + self._kinetic_energy(new_mom)
            valid = np.isfinite(new_h) and (new_h - start_h) <= 1000
            
            return {
            'position': new_pos,
            'momentum': new_mom,
            'leftmost_pos': new_pos,  # for single step, left=right
            'rightmost_pos': new_pos,
            'leftmost_mom': new_mom,
            'rightmost_mom': new_mom,
            'sum_prob': np.exp(start_h-new_h) if valid else 0.0,  # probability of this point
            'n_valid': valid,  # check if step was valid
            'uturn': False    # single step can't u-turn
            }
            
        else:
            left = self._build_tree(position, momentum, start_h, direction, depth-1)
            
            if left['uturn']:
                return left
            
            if direction == -1:
                right = self._build_tree(left['left_most_pos'], left['left_most_mom'], start_h, direction, depth-1)
            else:
                right = self._build_tree(left['right_most_pos'], left['right_most_mom'], start_h, direction, depth-1)
            
        
            combined_info = {
            'position': right['position'],
            'momentum': right['momentum'],
            'leftmost_pos': left['leftmost_pos'],
            'leftmost_mom': left['leftmost_mom'], 
            'rightmost_pos': right['rightmost_pos'],
            'rightmost_mom': right['rightmost_mom'],
            'sum_prob': left['sum_prob'] + right['sum_prob'],
            'n_valid': left['n_valid'] + right['n_valid'],
            }
        
            # check for U-turns
            combined_info['uturn'] = self._check_uturn(combined_info['leftmost_pos'], combined_info['rightmost_pos'], combined_info['leftmost_mom'], combined_info['rightmost_mom'])
            
            return combined_info
            



    def next_proposal(self, position, momentum):
        
        for _ in range(self.n_steps):
            position, momentum = self._leapfrog_step(position, momentum)
            
        return position, momentum
            
    def sample(self, current_position, n_samples, burn_in = 1000):
        samples = np.zeros((n_samples,dim))
        dim = len(current_position)
        
        
        for i in n_samples:
            momentum = np.random.normal(0, 1, dim)
            
            current_log_prob = self.log_prob_fn(current_position)
            current_kinetic_e = self._kinetic_energy(momentum)
            current_hamiltonian = -current_log_prob + current_kinetic_e
            
            tree = self. _build_tree(current_position, momentum, current_hamiltonian, 1, self.n_steps)
            
            if tree['valid']:
                
                
            
                
                                     
            '''
            next_position, next_momentum = self.next_proposal(current_position, momentum)
            
            next_log_prob = self.log_prob_fn(next_position)
            next_kinetic_e = self._kinetic_energy(next_momentum)
            next_hamiltonian = -next_log_prob+next_kinetic_e
            
            log_accaptance_ratio = current_hamiltonian-next_hamiltonian
            
            if log_accaptance_ratio > np.log(np.random.random()):
                
                current_position = next_position
            '''    
            
            samples[i] = current_position
            
        return samples[burn_in:]
    
    
    
    
    
    
    
    
    
    def _build_tree(self, position, momentum, start_h, direction, depth):
        
        if depth == 0:
            new_pos, new_mom = self._leapfrog_step(position, momentum, direction)
        
            new_h = -self.log_prob_fn(new_pos) + self._kinetic_energy(new_mom)
            n = 1 if start_h > new_h else 0
            valid = np.isfinite(new_h) and (new_h - start_h) <= 1000
            
            
            return {
            'position': new_pos,
            'leftmost_pos': new_pos,  # for single step, left=right
            'rightmost_pos': new_pos,
            'leftmost_mom': new_mom,
            'rightmost_mom': new_mom,
            'n': n,  # probability of this point
            'valid': valid,  # check if step was valid
            }
            
        else:
            left = self._build_tree(position, momentum, start_h, direction, depth-1)

            if left['stop']:
                return left
            
            if direction == -1:
                right = self._build_tree(left['left_most_pos'], left['left_most_mom'], start_h, direction, depth-1)
            else:
                right = self._build_tree(left['right_most_pos'], left['right_most_mom'], start_h, direction, depth-1)
        
            
            combined_info = {
            'position': right['position'] if scipy.stats.norm.pdf(1, 0, right['n']/(left['n'] + right['n'])) else left['position'],
            'leftmost_pos': left['leftmost_pos'],
            'leftmost_mom': left['leftmost_mom'], 
            'rightmost_pos': right['rightmost_pos'],
            'rightmost_mom': right['rightmost_mom'],
            'n': left['n'] + right['n'],
            }
            # check for U-turns
            combined_info['valid'] = self._check_uturn(combined_info['leftmost_pos'], combined_info['rightmost_pos'], combined_info['leftmost_mom'], combined_info['rightmost_mom'])
            
            return combined_info
            
            
            
            
        
        