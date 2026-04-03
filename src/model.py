import torch
import torch.nn as nn


class BioSimModel(nn.Module):
    """
    A simple feedforward neural network for intervention-aware multi-step biological state prediction.
    
    This model predicts how biological systems will evolve over multiple time steps,
    taking into account both the current state and any interventions (like drug effects).
    
    Input: concatenation of [current_state_vector, intervention_vector]
           The intervention vector represents external factors like treatments or drugs
           that can modify how the biological system behaves.
    
    Output: flattened future states (predicted states for the next N time steps)
    
    Example with 5 features and 3 future steps:
    - Input: 10 features (5 state + 5 intervention)
    - Output: 15 features (represents 3 future states: [x_{t+1}, x_{t+2}, x_{t+3}] flattened)
    
    The model learns to predict biological responses to interventions, which is useful
    for drug discovery, treatment planning, and understanding system dynamics.
    """
    
    def __init__(self, input_dim=10, output_dim=15, hidden_dim=32):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Number of features in the input (state + intervention)
                       e.g., for 5 state features + 5 intervention features: 10
            output_dim: Number of features in the output (num_features * future_steps)
                       e.g., for 5 features and 3 future steps: 5 * 3 = 15
            hidden_dim: Number of neurons in the hidden layers
        """
        super(BioSimModel, self).__init__()
        
        # First hidden layer: maps input_dim -> hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Second hidden layer: maps hidden_dim -> hidden_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer: maps hidden_dim -> output_dim (predict multiple future states)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Activation function (ReLU introduces non-linearity)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Contains both current state and intervention information
        
        Returns:
            output: Flattened future states of shape (batch_size, output_dim)
                   Represents [x_{t+1}, x_{t+2}, ..., x_{t+N}] concatenated
        """
        # Pass through first hidden layer and apply ReLU activation
        x = self.relu(self.fc1(x))
        
        # Pass through second hidden layer and apply ReLU activation
        x = self.relu(self.fc2(x))
        
        # Pass through output layer (no activation - raw prediction)
        x = self.fc3(x)
        
        return x
