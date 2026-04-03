import torch
import torch.nn as nn
import torch.optim as optim

from load_data import get_data_tensors
from model import BioSimModel


# Configuration
NUM_SAMPLES = 1000
NUM_FEATURES = 5
SEQUENCE_LENGTH = 100
FUTURE_STEPS = 3  # Predict the next 3 time steps
INTERVENTION_STRENGTH = 0.5  # How strongly interventions affect the system
HIDDEN_DIM = 32
EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MODEL_SAVE_PATH = "biosim_model.pt"


def main():
    """Train the BioSimModel for intervention-aware multi-step biological state prediction."""
    
    # === Step 1: Generate synthetic data with interventions ===
    print("Generating synthetic data with interventions...")
    x_input, x_future = get_data_tensors(
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES,
        sequence_length=SEQUENCE_LENGTH,
        future_steps=FUTURE_STEPS,
        intervention_strength=INTERVENTION_STRENGTH
    )
    print(f"Data shape:")
    print(f"  x_input (state + intervention): {x_input.shape}")
    print(f"  x_future (target): {x_future.shape}\n")
    
    # === Step 2: Create the model ===
    # Input includes both state and intervention vectors
    input_dim = NUM_FEATURES * 2  # state features + intervention features
    output_dim = NUM_FEATURES * FUTURE_STEPS  # flattened future states
    
    print("Creating BioSimModel...")
    model = BioSimModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=HIDDEN_DIM
    )
    print(f"Model:")
    print(f"  Input dim: {input_dim} ({NUM_FEATURES} state + {NUM_FEATURES} intervention)")
    print(f"  Output dim: {output_dim} ({NUM_FEATURES} features × {FUTURE_STEPS} future steps)")
    print(f"  Hidden dim: {HIDDEN_DIM}\n")
    print(f"Model architecture:\n{model}\n")
    
    # === Step 3: Define loss function and optimizer ===
    # MSE (Mean Squared Error) measures prediction accuracy
    criterion = nn.MSELoss()
    
    # Adam is a popular optimizer that adapts learning rates
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # === Step 4: Training loop ===
    print("Starting training...\n")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0
        
        # Process data in batches
        for i in range(0, x_input.shape[0], BATCH_SIZE):
            # Get a batch of data
            batch_input = x_input[i:i + BATCH_SIZE]  # state + intervention
            batch_future = x_future[i:i + BATCH_SIZE]  # target future states
            
            # Forward pass: predict future states given current state + intervention
            predictions = model(batch_input)
            
            # Calculate loss
            loss = criterion(predictions, batch_future)
            
            # Backward pass: compute gradients
            optimizer.zero_grad()  # Clear old gradients
            loss.backward()        # Compute new gradients
            optimizer.step()       # Update model parameters
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for the epoch
        avg_loss = epoch_loss / num_batches
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {avg_loss:.4f}")
    
    print("\nTraining complete!\n")
    
    # === Step 5: Save the trained model ===
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()

