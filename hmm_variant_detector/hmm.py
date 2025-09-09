# File: hmm_variant_detector/hmm.py
import numpy as np
from scipy.special import logsumexp
import warnings

class VariantHMM:
    """
    2-state, 2-channel Hidden Markov Model for A->G variant detection.
    
    States:
    - State 0: Reference (A) - high reference counts, low alternate counts
    - State 1: Variant (G) - balanced or high alternate counts
    
    Channels:
    - Channel 0: Reference allele counts
    - Channel 1: Alternate allele counts
    """
    
    def __init__(self, n_states=2, n_channels=2, random_state=None):
        """
        Initialize the HMM with default parameters.
        
        Parameters:
        -----------
        n_states : int
            Number of hidden states (default: 2)
        n_channels : int  
            Number of observation channels (default: 2)
        random_state : int, optional
            Random seed for reproducible results
        """
        self.n_states = n_states
        self.n_channels = n_channels
        
        if random_state is not None:
            np.random.seed(random_state)
            
        # Initialize parameters
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize HMM parameters with biologically reasonable defaults."""
        # Initial state probabilities (start with reference state more likely)
        self.pi = np.array([0.8, 0.2])
        
        # Transition matrix (tend to stay in same state)
        self.A = np.array([
            [0.95, 0.05],  # From reference to reference/variant
            [0.10, 0.90]   # From variant to reference/variant
        ])
        
        # Emission parameters for negative binomial distribution
        # State 0 (Reference): High ref counts, low alt counts
        # State 1 (Variant): More balanced counts
        self.emission_params = {
            'n': np.array([[50, 5], [20, 20]]),  # dispersion parameter
            'p': np.array([[0.1, 0.8], [0.3, 0.3]])  # success probability
        }
        
    def _negative_binomial_log_prob(self, x, n, p):
        """
        Compute log probability of negative binomial distribution.
        
        Parameters:
        -----------
        x : array_like
            Observed counts
        n : float
            Dispersion parameter
        p : float
            Success probability
            
        Returns:
        --------
        log_prob : float
            Log probability
        """
        from scipy.special import gammaln
        
        # Avoid numerical issues
        p = np.clip(p, 1e-10, 1-1e-10)
        
        log_prob = (gammaln(x + n) - gammaln(x + 1) - gammaln(n) + 
                   n * np.log(p) + x * np.log(1 - p))
        
        return log_prob
    
    def _emission_probability(self, observation, state):
        """
        Compute emission probability for a given observation and state.
        
        Parameters:
        -----------
        observation : array_like, shape (n_channels,)
            Observation vector (ref_count, alt_count)
        state : int
            Hidden state
            
        Returns:
        --------
        log_prob : float
            Log emission probability
        """
        log_prob = 0.0
        
        for channel in range(self.n_channels):
            n = self.emission_params['n'][state, channel]
            p = self.emission_params['p'][state, channel]
            
            log_prob += self._negative_binomial_log_prob(
                observation[channel], n, p
            )
            
        return log_prob
    
    def forward(self, observations):
        """
        Forward algorithm to compute observation likelihood.
        
        Parameters:
        -----------
        observations : array_like, shape (T, n_channels)
            Sequence of observations
            
        Returns:
        --------
        log_likelihood : float
            Log likelihood of observations
        alpha : array_like, shape (T, n_states)
            Forward probabilities
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialize
        for s in range(self.n_states):
            alpha[0, s] = (np.log(self.pi[s]) + 
                          self._emission_probability(observations[0], s))
        
        # Forward pass
        for t in range(1, T):
            for s in range(self.n_states):
                log_sum_terms = []
                for prev_s in range(self.n_states):
                    term = (alpha[t-1, prev_s] + 
                           np.log(self.A[prev_s, s]))
                    log_sum_terms.append(term)
                
                alpha[t, s] = (logsumexp(log_sum_terms) + 
                              self._emission_probability(observations[t], s))
        
        # Total likelihood
        log_likelihood = logsumexp(alpha[-1, :])
        
        return log_likelihood, alpha
    
    def backward(self, observations):
        """
        Backward algorithm to compute backward probabilities.
        
        Parameters:
        -----------
        observations : array_like, shape (T, n_channels)
            Sequence of observations
            
        Returns:
        --------
        beta : array_like, shape (T, n_states)
            Backward probabilities
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # Initialize (log probability 1 = 0)
        beta[T-1, :] = 0.0
        
        # Backward pass
        for t in range(T-2, -1, -1):
            for s in range(self.n_states):
                log_sum_terms = []
                for next_s in range(self.n_states):
                    term = (beta[t+1, next_s] + 
                           np.log(self.A[s, next_s]) +
                           self._emission_probability(observations[t+1], next_s))
                    log_sum_terms.append(term)
                
                beta[t, s] = logsumexp(log_sum_terms)
        
        return beta
    
    def viterbi(self, observations):
        """
        Viterbi algorithm to find most likely state sequence.
        
        Parameters:
        -----------
        observations : array_like, shape (T, n_channels)
            Sequence of observations
            
        Returns:
        --------
        states : array_like, shape (T,)
            Most likely state sequence
        log_prob : float
            Log probability of the sequence
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialize
        for s in range(self.n_states):
            delta[0, s] = (np.log(self.pi[s]) + 
                          self._emission_probability(observations[0], s))
        
        # Forward pass
        for t in range(1, T):
            for s in range(self.n_states):
                # Find best previous state
                scores = []
                for prev_s in range(self.n_states):
                    score = (delta[t-1, prev_s] + 
                           np.log(self.A[prev_s, s]))
                    scores.append(score)
                
                best_prev = np.argmax(scores)
                delta[t, s] = (scores[best_prev] + 
                              self._emission_probability(observations[t], s))
                psi[t, s] = best_prev
        
        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1, :])
        log_prob = delta[T-1, states[T-1]]
        
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states, log_prob
    
    def decode(self, observations):
        """
        Decode the most likely state sequence using Viterbi algorithm.
        
        Parameters:
        -----------
        observations : array_like, shape (T, n_channels)
            Sequence of observations
            
        Returns:
        --------
        states : array_like, shape (T,)
            Most likely state sequence
        """
        states, _ = self.viterbi(observations)
        return states
    
    def _compute_posteriors(self, observations, alpha, beta):
        """
        Compute posterior probabilities using forward-backward algorithm.
        
        Parameters:
        -----------
        observations : array_like, shape (T, n_channels)
            Sequence of observations
        alpha : array_like, shape (T, n_states)
            Forward probabilities
        beta : array_like, shape (T, n_states)
            Backward probabilities
            
        Returns:
        --------
        gamma : array_like, shape (T, n_states)
            State posterior probabilities
        xi : array_like, shape (T-1, n_states, n_states)
            Transition posterior probabilities
        """
        T = len(observations)
        
        # State posteriors
        gamma = alpha + beta
        # Normalize
        for t in range(T):
            gamma[t, :] -= logsumexp(gamma[t, :])
        gamma = np.exp(gamma)
        
        # Transition posteriors
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] + 
                                  np.log(self.A[i, j]) +
                                  self._emission_probability(observations[t+1], j) +
                                  beta[t+1, j])
            
            # Normalize
            xi[t] -= logsumexp(xi[t])
            xi[t] = np.exp(xi[t])
        
        return gamma, xi
    
    def _update_parameters(self, observations, gamma, xi):
        """
        Update HMM parameters using EM algorithm (M-step).
        
        Parameters:
        -----------
        observations : array_like, shape (T, n_channels)
            Sequence of observations
        gamma : array_like, shape (T, n_states)
            State posterior probabilities
        xi : array_like, shape (T-1, n_states, n_states)
            Transition posterior probabilities
        """
        T = len(observations)
        
        # Update initial state probabilities
        self.pi = gamma[0, :]
        
        # Update transition probabilities
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.A[i, j] = np.sum(xi[:, i, j]) / np.sum(gamma[:-1, i])
        
        # Update emission parameters using method of moments
        for s in range(self.n_states):
            for c in range(self.n_channels):
                # Weighted observations for this state and channel
                weights = gamma[:, s]
                obs_c = observations[:, c]
                
                # Weighted mean and variance
                mean_obs = np.sum(weights * obs_c) / np.sum(weights)
                var_obs = (np.sum(weights * (obs_c - mean_obs)**2) / 
                          np.sum(weights))
                
                # Method of moments for negative binomial
                # mean = n(1-p)/p, var = n(1-p)/p^2
                if var_obs > mean_obs:  # Overdispersed
                    p = mean_obs / var_obs
                    n = mean_obs * p / (1 - p)
                else:  # Use Poisson approximation
                    p = 0.5
                    n = 2 * mean_obs
                
                # Clip parameters to avoid numerical issues
                self.emission_params['p'][s, c] = np.clip(p, 0.01, 0.99)
                self.emission_params['n'][s, c] = max(n, 0.1)
    
    def _validate_parameters(self):
        """
        Validate and fix parameters if they become corrupted.
        """
        # Fix initial probabilities
        if np.any(np.isnan(self.pi)) or np.any(self.pi < 0):
            self.pi = np.array([0.8, 0.2])
        self.pi = np.clip(self.pi, 1e-10, 1.0)
        self.pi = self.pi / np.sum(self.pi)
        
        # Fix transition matrix
        for i in range(self.n_states):
            if np.any(np.isnan(self.A[i, :])) or np.any(self.A[i, :] < 0):
                if i == 0:
                    self.A[i, :] = [0.95, 0.05]
                else:
                    self.A[i, :] = [0.10, 0.90]
            self.A[i, :] = np.clip(self.A[i, :], 1e-10, 1.0)
            self.A[i, :] = self.A[i, :] / np.sum(self.A[i, :])
        
        # Fix emission parameters
        for s in range(self.n_states):
            for c in range(self.n_channels):
                n = self.emission_params['n'][s, c]
                p = self.emission_params['p'][s, c]
                
                if np.isnan(n) or np.isinf(n) or n <= 0:
                    if s == 0 and c == 0:  # Reference state, ref channel
                        self.emission_params['n'][s, c] = 20.0
                    elif s == 0 and c == 1:  # Reference state, alt channel
                        self.emission_params['n'][s, c] = 2.0
                    else:  # Variant state
                        self.emission_params['n'][s, c] = 10.0
                
                if np.isnan(p) or np.isinf(p) or p <= 0 or p >= 1:
                    if s == 0 and c == 0:  # Reference state, ref channel
                        self.emission_params['p'][s, c] = 0.2
                    elif s == 0 and c == 1:  # Reference state, alt channel
                        self.emission_params['p'][s, c] = 0.8
                    else:  # Variant state
                        self.emission_params['p'][s, c] = 0.4
                
                # Final safety clipping
                self.emission_params['n'][s, c] = float(np.clip(self.emission_params['n'][s, c], 0.1, 1000.0))
                self.emission_params['p'][s, c] = float(np.clip(self.emission_params['p'][s, c], 0.01, 0.99))

    def train(self, observations, max_iterations=100, tolerance=1e-6, verbose=False):
        """
        Train HMM parameters using Baum-Welch algorithm.
        
        Parameters:
        -----------
        observations : array_like, shape (T, n_channels)
            Training observations
        max_iterations : int
            Maximum number of EM iterations
        tolerance : float
            Convergence tolerance for log likelihood
        verbose : bool
            Print training progress
            
        Returns:
        --------
        log_likelihoods : list
            Log likelihood at each iteration
        """
        observations = np.asarray(observations)
        log_likelihoods = []
        
        for iteration in range(max_iterations):
            # Validate parameters before each iteration
            self._validate_parameters()
            
            # E-step: Forward-backward algorithm
            log_likelihood, alpha = self.forward(observations)
            beta = self.backward(observations)
            gamma, xi = self._compute_posteriors(observations, alpha, beta)
            
            # M-step: Update parameters
            self._update_parameters(observations, gamma, xi)
            
            log_likelihoods.append(log_likelihood)
            
            if verbose:
                print(f"Iteration {iteration + 1}: Log likelihood = {log_likelihood:.4f}")
            
            # Check convergence
            if (iteration > 0 and 
                abs(log_likelihoods[-1] - log_likelihoods[-2]) < tolerance):
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        # Final validation
        self._validate_parameters()
        
        return log_likelihoods
    
    def predict_proba(self, observations):
        """
        Predict state probabilities for observations.
        
        Parameters:
        -----------
        observations : array_like, shape (T, n_channels)
            Sequence of observations
            
        Returns:
        --------
        probabilities : array_like, shape (T, n_states)
            State probabilities
        """
        log_likelihood, alpha = self.forward(observations)
        beta = self.backward(observations)
        gamma, _ = self._compute_posteriors(observations, alpha, beta)
        
        return gamma
    
    def get_parameters(self):
        """
        Get current HMM parameters.
        
        Returns:
        --------
        params : dict
            Dictionary containing all HMM parameters
        """
        return {
            'pi': self.pi.copy(),
            'A': self.A.copy(),
            'emission_params': {
                'n': self.emission_params['n'].copy(),
                'p': self.emission_params['p'].copy()
            }
        }
    
    def set_parameters(self, params):
        """
        Set HMM parameters.
        
        Parameters:
        -----------
        params : dict
            Dictionary containing HMM parameters
        """
        self.pi = params['pi'].copy()
        self.A = params['A'].copy()
        self.emission_params = {
            'n': params['emission_params']['n'].copy(),
            'p': params['emission_params']['p'].copy()
        }