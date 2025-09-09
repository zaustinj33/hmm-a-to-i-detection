# File: examples/basic_usage.py
"""
Basic usage example of the HMM Variant Detector.
"""

import numpy as np
import matplotlib.pyplot as plt
from hmm_variant_detector import VariantHMM, generate_synthetic_data, evaluate_predictions

def main():
    print("HMM Variant Detector - Basic Usage Example")
    print("=" * 50)
    
    # Generate synthetic data
    print("1. Generating synthetic NGS data...")
    observations, true_states = generate_synthetic_data(
        n_positions=200,
        variant_rate=0.15,
        coverage_mean=25,
        coverage_std=8,
        random_state=42
    )
    
    print(f"Generated {len(observations)} positions")
    print(f"True variant rate: {np.mean(true_states):.3f}")
    print(f"Mean coverage: {np.mean(np.sum(observations, axis=1)):.1f}")
    
    # Initialize and train HMM
    print("\n2. Training HMM...")
    hmm = VariantHMM(random_state=42)
    
    log_likelihoods = hmm.train(
        observations, 
        max_iterations=50, 
        tolerance=1e-6,
        verbose=True
    )
    
    # Make predictions
    print("\n3. Making predictions...")
    predicted_states = hmm.decode(observations)
    predicted_probs = hmm.predict_proba(observations)
    
    # Evaluate performance
    print("\n4. Evaluating performance...")
    metrics = evaluate_predictions(
        true_states, 
        predicted_states, 
        predicted_probs[:, 1]
    )
    
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-score: {metrics['f1_score']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    if metrics.get('auc_roc'):
        print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    
    # Display some examples
    print("\n5. Example predictions:")
    print("Position | Ref | Alt | True | Pred | Prob")
    print("-" * 40)
    for i in range(min(20, len(observations))):
        ref_count, alt_count = observations[i]
        true_state = true_states[i]
        pred_state = predicted_states[i]
        prob_variant = predicted_probs[i, 1]
        
        print(f"{i:8d} | {ref_count:3d} | {alt_count:3d} | {true_state:4d} | "
              f"{pred_state:4d} | {prob_variant:.3f}")
    
    # Plot training progress
    try:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 2)
        plt.scatter(predicted_probs[:, 1], true_states, alpha=0.6)
        plt.xlabel('Predicted Variant Probability')
        plt.ylabel('True State')
        plt.title('Prediction vs Truth')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('hmm_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nPlots saved as 'hmm_results.png'")
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    
    # Show model parameters
    print("\n6. Learned HMM parameters:")
    params = hmm.get_parameters()
    print(f"Initial state probabilities: {params['pi']}")
    print("Transition matrix:")
    print(params['A'])
    print("Emission parameters (n): ")
    print(params['emission_params']['n'])
    print("Emission parameters (p): ")
    print(params['emission_params']['p'])

if __name__ == "__main__":
    main()
