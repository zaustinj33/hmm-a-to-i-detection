# File: tests/test_hmm.py
import pytest
import numpy as np
from hmm_variant_detector import VariantHMM, generate_synthetic_data, evaluate_predictions

class TestVariantHMM:
    
    def test_initialization(self):
        """Test HMM initialization."""
        hmm = VariantHMM()
        assert hmm.n_states == 2
        assert hmm.n_channels == 2
        assert hmm.pi.shape == (2,)
        assert hmm.A.shape == (2, 2)
        assert np.allclose(np.sum(hmm.pi), 1.0)
        assert np.allclose(np.sum(hmm.A, axis=1), 1.0)
    
    def test_forward_algorithm(self):
        """Test forward algorithm."""
        hmm = VariantHMM(random_state=42)
        observations = np.array([[10, 0], [8, 2], [5, 5]])
        
        log_likelihood, alpha = hmm.forward(observations)
        
        assert isinstance(log_likelihood, float)
        assert alpha.shape == (3, 2)
        assert not np.any(np.isnan(alpha))
        assert not np.any(np.isinf(alpha))
    
    def test_viterbi_algorithm(self):
        """Test Viterbi algorithm."""
        hmm = VariantHMM(random_state=42)
        observations = np.array([[10, 0], [8, 2], [5, 5]])
        
        states, log_prob = hmm.viterbi(observations)
        
        assert states.shape == (3,)
        assert isinstance(log_prob, float)
        assert np.all((states == 0) | (states == 1))
    
    def test_decode_method(self):
        """Test decode method."""
        hmm = VariantHMM(random_state=42)
        observations = np.array([[10, 0], [8, 2], [5, 5]])
        
        states = hmm.decode(observations)
        
        assert states.shape == (3,)
        assert np.all((states == 0) | (states == 1))
    
    def test_training(self):
        """Test Baum-Welch training."""
        # Generate synthetic data
        observations, true_states = generate_synthetic_data(
            n_positions=100, random_state=42
        )
        
        hmm = VariantHMM(random_state=42)
        
        # Train the model
        log_likelihoods = hmm.train(observations, max_iterations=10, verbose=False)
        
        assert len(log_likelihoods) <= 10
        assert all(isinstance(ll, float) for ll in log_likelihoods)
        
        # Check that likelihood generally increases
        if len(log_likelihoods) > 1:
            assert log_likelihoods[-1] >= log_likelihoods[0] - 1.0  # Allow some tolerance
    
    def test_prediction(self):
        """Test prediction methods."""
        # Generate and train on synthetic data
        observations, true_states = generate_synthetic_data(
            n_positions=50, random_state=42
        )
        
        hmm = VariantHMM(random_state=42)
        hmm.train(observations, max_iterations=10, verbose=False)
        
        # Test predictions
        predicted_states = hmm.decode(observations)
        predicted_probs = hmm.predict_proba(observations)
        
        assert predicted_states.shape == (50,)
        assert predicted_probs.shape == (50, 2)
        assert np.allclose(np.sum(predicted_probs, axis=1), 1.0)
        
        # Evaluate performance
        metrics = evaluate_predictions(true_states, predicted_states, 
                                     predicted_probs[:, 1])
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

def test_generate_synthetic_data():
    """Test synthetic data generation."""
    observations, true_states = generate_synthetic_data(
        n_positions=100, variant_rate=0.2, random_state=42
    )
    
    assert observations.shape == (100, 2)
    assert true_states.shape == (100,)
    assert np.all((true_states == 0) | (true_states == 1))
    assert np.all(observations >= 0)
    
    # Check variant rate is approximately correct
    variant_fraction = np.mean(true_states)
    assert 0.1 <= variant_fraction <= 0.3  # Should be around 0.2

def test_evaluate_predictions():
    """Test evaluation metrics."""
    true_states = np.array([0, 0, 1, 1, 0, 1])
    predicted_states = np.array([0, 1, 1, 1, 0, 0])
    predicted_probs = np.array([0.1, 0.8, 0.9, 0.7, 0.2, 0.4])
    
    metrics = evaluate_predictions(true_states, predicted_states, predicted_probs)
    
    required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 
                       'true_positives', 'false_positives', 
                       'true_negatives', 'false_negatives']
    
    for metric in required_metrics:
        assert metric in metrics

if __name__ == "__main__":
    pytest.main([__file__])