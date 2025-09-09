# File: hmm_variant_detector/utils.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def generate_synthetic_data(n_positions=1000, variant_rate=0.1, coverage_mean=30, 
                          coverage_std=10, random_state=None):
    """
    Generate synthetic NGS data with A->G variants.
    
    Parameters:
    -----------
    n_positions : int
        Number of genomic positions
    variant_rate : float
        Proportion of positions with variants
    coverage_mean : float
        Mean coverage depth
    coverage_std : float
        Standard deviation of coverage
    random_state : int, optional
        Random seed
        
    Returns:
    --------
    observations : array_like, shape (n_positions, 2)
        Synthetic observations [ref_counts, alt_counts]
    true_states : array_like, shape (n_positions,)
        True variant states (0=reference, 1=variant)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate true states
    true_states = np.random.binomial(1, variant_rate, n_positions)
    
    observations = np.zeros((n_positions, 2), dtype=int)
    
    for i in range(n_positions):
        # Generate coverage
        coverage = max(1, int(np.random.normal(coverage_mean, coverage_std)))
        
        if true_states[i] == 0:  # Reference position
            # Mostly reference alleles, few alternate (sequencing errors)
            alt_prob = 0.02  # 2% error rate
            alt_count = np.random.binomial(coverage, alt_prob)
            ref_count = coverage - alt_count
        else:  # Variant position
            # More balanced or alternate-heavy
            alt_prob = np.random.uniform(0.3, 0.8)  # Heterozygous to homozygous
            alt_count = np.random.binomial(coverage, alt_prob)
            ref_count = coverage - alt_count
        
        observations[i] = [ref_count, alt_count]
    
    return observations, true_states

def evaluate_predictions(true_states, predicted_states, predicted_probs=None):
    """
    Evaluate variant detection performance.
    
    Parameters:
    -----------
    true_states : array_like
        True variant states
    predicted_states : array_like
        Predicted variant states
    predicted_probs : array_like, optional
        Predicted probabilities for variant state
        
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(true_states, predicted_states)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        true_states, predicted_states, average='binary'
    )
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    metrics['support'] = support
    
    # Confusion matrix components
    tp = np.sum((true_states == 1) & (predicted_states == 1))
    tn = np.sum((true_states == 0) & (predicted_states == 0))
    fp = np.sum((true_states == 0) & (predicted_states == 1))
    fn = np.sum((true_states == 1) & (predicted_states == 0))
    
    metrics['true_positives'] = tp
    metrics['true_negatives'] = tn
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    
    # Specificity
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    if predicted_probs is not None:
        from sklearn.metrics import roc_auc_score
        try:
            metrics['auc_roc'] = roc_auc_score(true_states, predicted_probs)
        except ValueError:
            metrics['auc_roc'] = None
    
    return metrics