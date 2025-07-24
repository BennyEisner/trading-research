#!/usr/bin/env python3

"""
Simple hyperparameter fallback for zero temporal loss training
"""

def get_optimized_hyperparameters_for_large_dataset():
    """
    Return optimized hyperparameters for large financial dataset (18K+ samples)
    Based on financial ML best practices and large dataset considerations
    """
    
    return {
        # Directional loss configuration
        'directional_alpha': 0.35,  # Balanced focus on direction vs magnitude
        
        # Learning parameters optimized for large dataset
        'learning_rate': 0.0008,  # Slightly higher for faster convergence
        'batch_size': 128,        # Larger batches for efficiency with large dataset
        
        # Regularization for large dataset
        'dropout_rate': 0.3,      # Lower dropout with more data
        'l2_regularization': 0.003,  # Reduced regularization with more samples
        
        # Architecture scaled for financial complexity
        'lstm_units_1': 512,      # Primary pattern recognition
        'lstm_units_2': 256,      # Pattern refinement  
        'lstm_units_3': 128,      # Final pattern extraction
        'dense_layers': [256, 128, 64],  # Deep feature fusion
        
        # Advanced features
        'use_attention': True,    # Enable attention mechanisms
        'recurrent_dropout': 0.2, # Moderate recurrent dropout
        'activity_regularizer': 0.0003,  # Light activity regularization
        
        # Training configuration
        'patience': 20,           # More patience with large dataset
        'max_epochs': 150,        # More epochs for complex patterns
        'validation_split': 0.15, # Smaller val split with large dataset
        
        # Model ensemble
        'ensemble_size': 1,       # Single model for now
        'model_averaging': False
    }


def get_directional_training_config():
    """
    Return training configuration optimized for directional accuracy
    """
    
    return {
        # Callbacks optimized for directional learning
        'early_stopping': {
            'monitor': 'val__directional_accuracy',
            'patience': 20,
            'restore_best_weights': True,
            'mode': 'max',
            'min_delta': 0.001
        },
        
        'reduce_lr': {
            'monitor': 'val__directional_accuracy',
            'factor': 0.5,
            'patience': 10,
            'mode': 'max',
            'min_lr': 1e-6,
            'verbose': 1
        },
        
        'model_checkpoint': {
            'monitor': 'val__directional_accuracy',
            'save_best_only': True,
            'mode': 'max',
            'filepath': 'models/trained/best_zero_temporal_loss_model.keras'
        }
    }


def validate_hyperparameters(params):
    """
    Validate hyperparameter values for consistency
    """
    
    validations = []
    
    # Check directional alpha range
    if not 0.1 <= params.get('directional_alpha', 0.35) <= 0.8:
        validations.append("directional_alpha should be between 0.1 and 0.8")
    
    # Check learning rate range
    lr = params.get('learning_rate', 0.0008)
    if not 0.0001 <= lr <= 0.01:
        validations.append("learning_rate should be between 0.0001 and 0.01")
    
    # Check batch size
    batch_size = params.get('batch_size', 128)
    if batch_size not in [32, 64, 96, 128, 256]:
        validations.append("batch_size should be one of [32, 64, 96, 128, 256]")
    
    # Check architecture consistency
    units = [params.get('lstm_units_1', 512), params.get('lstm_units_2', 256), params.get('lstm_units_3', 128)]
    if not (units[0] >= units[1] >= units[2]):
        validations.append("LSTM units should be decreasing: units_1 >= units_2 >= units_3")
    
    return validations


def print_hyperparameter_summary(params):
    """
    Print a summary of the hyperparameters being used
    """
    
    print("OPTIMIZED HYPERPARAMETERS FOR ZERO TEMPORAL LOSS TRAINING")
    print("=" * 65)
    
    print("\nDirectional Loss Configuration:")
    print(f"  Directional Alpha: {params.get('directional_alpha', 0.35):.3f}")
    print(f"  Focus: {100 * params.get('directional_alpha', 0.35):.1f}% direction, {100 * (1 - params.get('directional_alpha', 0.35)):.1f}% magnitude")
    
    print("\n⚡ Learning Configuration:")
    print(f"  Learning Rate: {params.get('learning_rate', 0.0008):.6f}")
    print(f"  Batch Size: {params.get('batch_size', 128)}")
    print(f"  Max Epochs: {params.get('max_epochs', 150)}")
    
    print("\nArchitecture Configuration:")
    print(f"  LSTM Units: {params.get('lstm_units_1', 512)} → {params.get('lstm_units_2', 256)} → {params.get('lstm_units_3', 128)}")
    print(f"  Dense Layers: {params.get('dense_layers', [256, 128, 64])}")
    print(f"  Attention: {'Enabled' if params.get('use_attention', True) else 'Disabled'}")
    
    print("\n Regularization:")
    print(f"  Dropout Rate: {params.get('dropout_rate', 0.3):.3f}")
    print(f"  L2 Regularization: {params.get('l2_regularization', 0.003):.6f}")
    print(f"  Recurrent Dropout: {params.get('recurrent_dropout', 0.2):.3f}")
    
    # Validate parameters
    validation_issues = validate_hyperparameters(params)
    if validation_issues:
        print("\n Parameter Validation Issues:")
        for issue in validation_issues:
            print(f"  - {issue}")
    else:
        print("\n All hyperparameters validated successfully")
    
    print("\n Optimized for:")
    print("  - Large dataset (18,000+ training samples)")
    print("  - Financial time series directional prediction")
    print("  - Multi-scale temporal pattern recognition")
    print("  - Zero temporal loss data utilization")
    

if __name__ == "__main__":
    # Example usage
    params = get_optimized_hyperparameters_for_large_dataset()
    print_hyperparameter_summary(params)