# File: examples/advanced_usage.py
"""
Advanced usage example with custom data and parameter tuning.
"""

import numpy as np
from hmm_variant_detector import VariantHMM, evaluate_predictions

def load_custom_data():
    """
    Example function to load custom NGS data.
    Replace this with your actual data loading logic.
    """
    # This is a placeholder - replace with actual data loading
    # Expected format: array of shape (n_positions, 2) where
    # column 0 = reference allele counts, column 1 = alternate allele counts
    
    # For demonstration, we'll simulate some realistic data
    np.random.seed(123)
    n_positions = 500
    
    observations = []
    true_states = []
    
    for i in range(n_positions):
        # Simulate varying coverage
        coverage = np.random.poisson(30)
        coverage = max(5, coverage)  # Minimum coverage
        
        # Simulate variant status
        is_variant = np.random.random() < 0.12  # 12% variant rate
        
        if is_variant:
            # Variant position - more alternate alleles
            vaf = np.random.beta(2, 2)  # Variant allele frequency
            vaf = np.clip(vaf, 0.2, 0.8)  # Realistic range
            alt_count = np.random.binomial(coverage, vaf)
        else:
            # Reference position - mostly reference alleles
            error_rate = np.random.beta(1, 50)  # Low error rate
            alt_count = np.random.binomial(coverage, error_rate)
        
        ref_count = coverage - alt_count
        
        observations.append([ref_count, alt_count])
        true_states.append(int(is_variant))
    
    return np.array(observations), np.array(true_states)

def optimize_hyperparameters(train_obs, train_states, val_obs, val_states):
    """
    Simple hyperparameter optimization example.
    """
    print("Optimizing hyperparameters...")
    
    best_f1 = 0
    best_params = None
    
    # Try different random seeds for initialization
    for seed in [42, 123, 456, 789, 999]:
        hmm = VariantHMM(random_state=seed)
        
        # Train model
        hmm.train(train_obs, max_iterations=30, verbose=False)
        
        # Evaluate on validation set
        pred_states = hmm.decode(val_obs)
        pred_probs = hmm.predict_proba(val_obs)
        
        metrics = evaluate_predictions(val_states, pred_states, pred_probs[:, 1])
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_params = hmm.get_parameters()
            print(f"New best F1: {best_f1:.3f} (seed: {seed})")
    
    return best_params, best_f1

def analyze_difficult_cases(observations, true_states, predicted_states, predicted_probs):
    """
    Analyze cases where the model struggled.
    """
    print("\nAnalyzing difficult cases...")
    
    # Find false positives and false negatives
    fp_mask = (true_states == 0) & (predicted_states == 1)
    fn_mask = (true_states == 1) & (predicted_states == 0)
    
    print(f"False positives: {np.sum(fp_mask)}")
    print(f"False negatives: {np.sum(fn_mask)}")
    
    if np.sum(fp_mask) > 0:
        print("\nFalse positive examples:")
        fp_indices = np.where(fp_mask)[0][:5]  # Show first 5
        for idx in fp_indices:
            ref, alt = observations[idx]
            prob = predicted_probs[idx, 1]
            vaf = alt / (ref + alt) if (ref + alt) > 0 else 0
            print(f"  Position {idx}: {ref}R/{alt}A (VAF={vaf:.3f}, P={prob:.3f})")
    
    if np.sum(fn_mask) > 0:
        print("\nFalse negative examples:")
        fn_indices = np.where(fn_mask)[0][:5]  # Show first 5
        for idx in fn_indices:
            ref, alt = observations[idx]
            prob = predicted_probs[idx, 1]
            vaf = alt / (ref + alt) if (ref + alt) > 0 else 0
            print(f"  Position {idx}: {ref}R/{alt}A (VAF={vaf:.3f}, P={prob:.3f})")

def main():
    print("HMM Variant Detector - Advanced Usage Example")
    print("=" * 55)
    
    # Load custom data
    print("1. Loading custom data...")
    observations, true_states = load_custom_data()
    
    print(f"Loaded {len(observations)} positions")
    print(f"Variant rate: {np.mean(true_states):.3f}")
    
    # Split data for training and validation
    split_point = int(0.7 * len(observations))
    train_obs, val_obs = observations[:split_point], observations[split_point:]
    train_states, val_states = true_states[:split_point], true_states[split_point:]
    
    print(f"Training set: {len(train_obs)} positions")
    print(f"Validation set: {len(val_obs)} positions")
    
    # Optimize hyperparameters
    print("\n2. Hyperparameter optimization...")
    best_params, best_f1 = optimize_hyperparameters(
        train_obs, train_states, val_obs, val_states
    )
    
    # Train final model with best parameters
    print(f"\n3. Training final model (best F1: {best_f1:.3f})...")
    final_hmm = VariantHMM(random_state=42)
    final_hmm.set_parameters(best_params)
    
    # Fine-tune on full training set
    log_likelihoods = final_hmm.train(
        train_obs, 
        max_iterations=50, 
        tolerance=1e-6,
        verbose=True
    )
    
    # Evaluate on validation set
    print("\n4. Final evaluation...")
    val_pred_states = final_hmm.decode(val_obs)
    val_pred_probs = final_hmm.predict_proba(val_obs)
    
    val_metrics = evaluate_predictions(
        val_states, val_pred_states, val_pred_probs[:, 1]
    )
    
    print("Validation metrics:")
    for metric, value in val_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")
    
    # Analyze difficult cases
    analyze_difficult_cases(
        val_obs, val_states, val_pred_states, val_pred_probs
    )
    
    # Compare different algorithms
    print("\n5. Algorithm comparison...")
    
    # Forward algorithm likelihood
    forward_ll, _ = final_hmm.forward(val_obs)
    print(f"Forward algorithm log-likelihood: {forward_ll:.2f}")
    
    # Viterbi path probability
    viterbi_states, viterbi_ll = final_hmm.viterbi(val_obs)
    print(f"Viterbi path log-probability: {viterbi_ll:.2f}")
    
    # Compare Viterbi vs Forward-Backward decoding
    fb_probs = final_hmm.predict_proba(val_obs)
    fb_states = np.argmax(fb_probs, axis=1)
    
    agreement = np.mean(viterbi_states == fb_states)
    print(f"Viterbi vs Forward-Backward agreement: {agreement:.3f}")
    
    # Performance comparison
    viterbi_metrics = evaluate_predictions(val_states, viterbi_states)
    fb_metrics = evaluate_predictions(val_states, fb_states)
    
    print(f"Viterbi F1-score: {viterbi_metrics['f1_score']:.3f}")
    print(f"Forward-Backward F1-score: {fb_metrics['f1_score']:.3f}")
    
    # Coverage and VAF analysis
    print("\n6. Coverage and VAF analysis...")
    coverages = np.sum(val_obs, axis=1)
    vafs = val_obs[:, 1] / coverages
    vafs[coverages == 0] = 0  # Handle zero coverage
    
    # Analyze performance by coverage
    low_cov = coverages < 15
    high_cov = coverages >= 15
    
    if np.sum(low_cov) > 0:
        low_cov_metrics = evaluate_predictions(
            val_states[low_cov], val_pred_states[low_cov]
        )
        print(f"Low coverage (<15x) F1-score: {low_cov_metrics['f1_score']:.3f}")
    
    if np.sum(high_cov) > 0:
        high_cov_metrics = evaluate_predictions(
            val_states[high_cov], val_pred_states[high_cov]
        )
        print(f"High coverage (≥15x) F1-score: {high_cov_metrics['f1_score']:.3f}")
    
    # Analyze performance by VAF for true variants
    variant_mask = val_states == 1
    if np.sum(variant_mask) > 0:
        variant_vafs = vafs[variant_mask]
        variant_preds = val_pred_states[variant_mask]
        
        low_vaf = variant_vafs < 0.3
        high_vaf = variant_vafs >= 0.3
        
        if np.sum(low_vaf) > 0:
            low_vaf_recall = np.mean(variant_preds[low_vaf])
            print(f"Low VAF (<0.3) recall: {low_vaf_recall:.3f}")
        
        if np.sum(high_vaf) > 0:
            high_vaf_recall = np.mean(variant_preds[high_vaf])
            print(f"High VAF (≥0.3) recall: {high_vaf_recall:.3f}")

if __name__ == "__main__":
    main()