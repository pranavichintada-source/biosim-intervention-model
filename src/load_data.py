import numpy as np
import torch
import pandas as pd


def generate_synthetic_data(num_samples=1000, num_features=5, sequence_length=100, intervention_strength=0.5):
    """
    Generate synthetic biological time-series data with interventions.
    
    Args:
        num_samples: Number of independent time series to generate
        num_features: Number of features per sample
        sequence_length: Length of each time series
        intervention_strength: How strongly the intervention affects future states (0-1)
    
    Returns:
        data: ndarray of shape (num_samples, sequence_length, num_features)
              Simulates biological measurements over time
        interventions: ndarray of shape (num_samples, num_features)
                      Random intervention vectors (like drug effects) for each sample
    """
    data = np.zeros((num_samples, sequence_length, num_features))
    interventions = np.zeros((num_samples, num_features))
    
    for sample_idx in range(num_samples):
        # Generate a random intervention vector for this sample
        # This represents something like a drug effect or treatment
        interventions[sample_idx] = np.random.randn(num_features) * 0.5
        
        # Initialize with random starting values
        state = np.random.randn(num_features)
        
        for t in range(sequence_length):
            # Store current state
            data[sample_idx, t] = state
            
            # Update state with biological-like dynamics
            # Add damping and noise to simulate realistic behavior
            momentum = 0.9 * state
            noise = 0.1 * np.random.randn(num_features)
            state = momentum + noise
            
            # Apply intervention effect to future states (starting from t+1)
            # The intervention gradually affects the system
            if t >= 0:  # Apply from the beginning, but effect grows
                intervention_effect = interventions[sample_idx] * intervention_strength * (t + 1) / sequence_length
                state = state + intervention_effect
    
    return data, interventions


def get_data_tensors(num_samples=1000, num_features=5, sequence_length=100, future_steps=3, intervention_strength=0.5):
    """
    Generate synthetic data with interventions and return as (input, target) tensor pairs.
    
    Supports multi-step prediction with interventions: given the current state and
    intervention, predict the next future_steps states. The target future states are
    flattened into a single vector for easier neural network prediction.
    
    Input = concatenation of [current_state, intervention_vector]
    Target = flattened [x_{t+1}, x_{t+2}, ..., x_{t+future_steps}]
    
    Args:
        num_samples: Number of independent time series
        num_features: Number of features per sample
        sequence_length: Length of each time series
        future_steps: Number of future time steps to predict (default: 3)
        intervention_strength: How strongly interventions affect states (0-1)
    
    Returns:
        x_input: PyTorch tensor of shape (num_total_pairs, num_features * 2)
                 Concatenation of [current_state, intervention_vector]
        x_future: PyTorch tensor of shape (num_total_pairs, num_features * future_steps)
                  Represents flattened [x_{t+1}, x_{t+2}, ..., x_{t+future_steps}]
    """
    # Generate synthetic data with interventions
    data, interventions = generate_synthetic_data(
        num_samples, num_features, sequence_length, intervention_strength
    )
    
    # Create pairs of (input, future_states_flattened)
    x_input_list = []
    x_future_list = []
    
    for sample_idx in range(num_samples):
        # For each sample, create training pairs
        # We can only create pairs up to (sequence_length - future_steps)
        # because we need future_steps steps ahead to exist
        for t in range(sequence_length - future_steps):
            # Current state at time t
            current_state = data[sample_idx, t]
            
            # Intervention vector for this sample
            intervention = interventions[sample_idx]
            
            # Concatenate state and intervention as input
            input_vector = np.concatenate([current_state, intervention])
            x_input_list.append(input_vector)
            
            # Collect the next future_steps states
            future_states = []
            for step in range(1, future_steps + 1):
                future_states.append(data[sample_idx, t + step])
            
            # Flatten the future states into one vector
            # Example: 3 steps of 5 features each -> one vector of 15 features
            flattened_future = np.concatenate(future_states)
            x_future_list.append(flattened_future)
    
    # Convert to PyTorch tensors
    x_input = torch.tensor(np.array(x_input_list), dtype=torch.float32)
    x_future = torch.tensor(np.array(x_future_list), dtype=torch.float32)
    
    return x_input, x_future


def load_real_data(filepath, future_steps=3, num_features=None):
    """
    Load real biological time-series data from a CSV file.
    
    PLACEHOLDER FUNCTION: This is a template for integrating real biological data.
    Users should modify this function based on their specific data format and structure.
    
    Expected CSV format:
    - Rows: time points in a biological time series
    - Columns: biological features/measurements (gene expression, protein levels, etc.)
    
    Args:
        filepath: Path to the CSV file containing time-series data
        future_steps: Number of future time steps to predict (default: 3)
        num_features: Number of expected features (if None, inferred from data)
    
    Returns:
        x_input: PyTorch tensor of shape (num_pairs, num_features * 2)
                 Contains [current_state, zero_intervention] pairs
                 (Zero intervention used as placeholder for compatibility)
        x_future: PyTorch tensor of shape (num_pairs, num_features * future_steps)
                  Contains future states as flattened vectors
    
    Note:
        This function currently assumes:
        - Data is a single time series (not multiple independent samples)
        - All columns are numeric features
        - Data is clean (no missing values)
        
        For your specific data, you may need to:
        - Handle missing values (NaN)
        - Select specific columns
        - Normalize/standardize features
        - Handle multiple time series
    """
    # Load CSV file using pandas
    print(f"Loading real data from {filepath}...")
    data = pd.read_csv(filepath)
    
    # Extract numeric columns only
    numeric_data = data.select_dtypes(include=[np.number]).values
    
    print(f"Data shape: {numeric_data.shape}")
    print(f"Features: {numeric_data.shape[1]}")
    
    # Infer number of features if not provided
    if num_features is None:
        num_features = numeric_data.shape[1]
    
    # Create time-lagged pairs from the time series
    # Assume rows are ordered time points
    x_input_list = []
    x_future_list = []
    
    for t in range(numeric_data.shape[0] - future_steps):
        # Current state at time t
        current_state = numeric_data[t, :num_features]
        
        # For now, use zero intervention as placeholder
        # In real applications, this could be:
        # - An intervention vector from another column
        # - Treatment information from metadata
        # - A learned intervention representation
        zero_intervention = np.zeros(num_features)
        
        # Concatenate state and intervention
        input_vector = np.concatenate([current_state, zero_intervention])
        x_input_list.append(input_vector)
        
        # Collect next future_steps states
        future_states = []
        for step in range(1, future_steps + 1):
            if t + step < numeric_data.shape[0]:
                future_states.append(numeric_data[t + step, :num_features])
        
        # Only add if we have all future steps
        if len(future_states) == future_steps:
            flattened_future = np.concatenate(future_states)
            x_future_list.append(flattened_future)
    
    print(f"Created {len(x_input_list)} training pairs\n")
    
    # Convert to PyTorch tensors
    x_input = torch.tensor(np.array(x_input_list), dtype=torch.float32)
    x_future = torch.tensor(np.array(x_future_list), dtype=torch.float32)
    
    return x_input, x_future
