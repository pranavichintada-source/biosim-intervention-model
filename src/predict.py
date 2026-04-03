import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from load_data import get_data_tensors, load_real_data
from model import BioSimModel


# Configuration
# === Data Source Selection ===
use_real_data = False  # Set to True to load from CSV, False for synthetic data
real_data_path = "data/sample_real_data.csv"  # Path to real data CSV file

# === Model Parameters ===
NUM_FEATURES = 5
HIDDEN_DIM = 32
FUTURE_STEPS = 3  # Predict the next 3 time steps
INTERVENTION_STRENGTH = 0.5  # How strongly interventions affect the system

# Resolve the model path relative to this script, so it works regardless of cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "biosim_model.pt"))

EXAMPLE_INDEX = 0  # Choose which sample to use
PLOT_FEATURE_INDEX = 0  # Which feature to plot (0 to NUM_FEATURES-1)


def main():
    """Make intervention-aware multi-step predictions using the trained BioSimModel."""
    
    # === Step 1: Load the trained model ===
    print("Loading trained model...\n")
    print(f"Resolved model path: {MODEL_PATH}\n")
    input_dim = NUM_FEATURES * 2  # state + intervention features
    output_dim = NUM_FEATURES * FUTURE_STEPS
    model = BioSimModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=HIDDEN_DIM
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)
    
    # === Step 2: Load data (synthetic or real) ===
    if use_real_data:
        # Load real biological data from CSV file
        print("Loading real data from CSV...\n")
        try:
            x_input, x_future = load_real_data(
                filepath=real_data_path,
                future_steps=FUTURE_STEPS
            )
        except FileNotFoundError:
            print(f"Error: Could not find {real_data_path}")
            print("Make sure the CSV file exists at the specified path.")
            print("Falling back to synthetic data...\n")
            x_input, x_future = get_data_tensors(
                num_samples=100,
                num_features=NUM_FEATURES,
                sequence_length=50,
                future_steps=FUTURE_STEPS,
                intervention_strength=INTERVENTION_STRENGTH
            )
    else:
        # Generate synthetic data (default)
        print("Generating synthetic data with interventions...")
        x_input, x_future = get_data_tensors(
            num_samples=100,
            num_features=NUM_FEATURES,
            sequence_length=50,
            future_steps=FUTURE_STEPS,
            intervention_strength=INTERVENTION_STRENGTH
        )
    
    print(f"Sample data shape: x_input={x_input.shape}, x_future={x_future.shape}\n")
    
    # === Step 3: Select one example ===
    print(f"Using example index: {EXAMPLE_INDEX}\n")
    input_combined = x_input[EXAMPLE_INDEX:EXAMPLE_INDEX + 1]  # Keep batch dimension
    true_future_flat = x_future[EXAMPLE_INDEX:EXAMPLE_INDEX + 1]
    
    # Separate the input into state and intervention components
    current_state = input_combined[:, :NUM_FEATURES]  # First half: state
    intervention = input_combined[:, NUM_FEATURES:]   # Second half: intervention
    
    # === Step 4: Make prediction ===
    with torch.no_grad():  # Disable gradient computation (faster, less memory)
        predicted_future_flat = model(input_combined)
    
    # === Step 5: Reshape flattened targets for easy reading ===
    # Convert from shape (1, num_features * future_steps) to (future_steps, num_features)
    true_future = true_future_flat.squeeze().numpy().reshape(FUTURE_STEPS, NUM_FEATURES)
    predicted_future = predicted_future_flat.squeeze().numpy().reshape(FUTURE_STEPS, NUM_FEATURES)
    
    # === Step 6: Display results ===
    print("=" * 80)
    print("INTERVENTION-AWARE MULTI-STEP PREDICTION RESULTS")
    print("=" * 80)
    
    print("\nCurrent State (x_t):")
    print(current_state.squeeze().numpy())
    
    print("\nIntervention Vector:")
    print(intervention.squeeze().numpy())
    print("(This represents external factors like drug effects)")
    
    print("\n" + "-" * 80)
    print("Future States (next {} time steps):".format(FUTURE_STEPS))
    print("-" * 80)
    
    for step in range(FUTURE_STEPS):
        print(f"\nStep t+{step+1}:")
        print(f"  True:      {true_future[step]}")
        print(f"  Predicted: {predicted_future[step]}")
        
        # Calculate error for this step
        error = np.abs(predicted_future[step] - true_future[step])
        mean_error = error.mean()
        print(f"  Error:     {error}")
        print(f"  MAE:       {mean_error:.4f}")
    
    # === Step 7: Calculate overall prediction error ===
    print("\n" + "=" * 80)
    overall_error = np.abs(predicted_future - true_future)
    overall_mae = overall_error.mean()
    print(f"Overall Mean Absolute Error (all steps): {overall_mae:.4f}")
    print("=" * 80)
    
    # === Step 8: Create visualization ===
    print("\nCreating visualization...")
    
    # Extract the chosen feature (PLOT_FEATURE_INDEX) for each future step
    true_values = true_future[:, PLOT_FEATURE_INDEX]  # Shape: (FUTURE_STEPS,)
    predicted_values = predicted_future[:, PLOT_FEATURE_INDEX]  # Shape: (FUTURE_STEPS,)
    
    # Create time steps for x-axis (t+1, t+2, t+3, etc.)
    time_steps = [f"t+{i+1}" for i in range(FUTURE_STEPS)]
    
    # Create the plot
    plt.figure(figsize=(8, 5))  # Set figure size
    
    # Plot true values
    plt.plot(time_steps, true_values, 'b-o', label='True', linewidth=2, markersize=6)
    
    # Plot predicted values
    plt.plot(time_steps, predicted_values, 'r--s', label='Predicted', linewidth=2, markersize=6)
    
    # Add labels and title
    plt.xlabel('Future Time Steps')
    plt.ylabel(f'Feature {PLOT_FEATURE_INDEX} Value')
    plt.title(f'BioSim Intervention-Aware Prediction: Feature {PLOT_FEATURE_INDEX}')
    plt.legend()
    plt.grid(True, alpha=0.3)  # Add light grid
    
    # Save the plot before showing it
    assets_dir = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'assets'))
    os.makedirs(assets_dir, exist_ok=True)
    plot_path = os.path.join(assets_dir, 'prediction_plot.png')
    plt.savefig(plot_path)
    print(f"Saved prediction plot to: {plot_path}")
    
    # Show the plot
    plt.show(block=False)
    
    # === Step 9: Intervention Comparison Experiment ===
    print("\n" + "=" * 80)
    print("INTERVENTION COMPARISON EXPERIMENT")
    print("=" * 80)
    print("\nHow does the same biological state evolve under different intervention conditions?")
    print()
    
    # Create two intervention scenarios for the same current state
    # Scenario 1: Zero intervention (no external factors)
    intervention_zero = np.zeros_like(intervention.squeeze().numpy())
    input_no_intervention = torch.cat([current_state, torch.tensor(intervention_zero, dtype=torch.float32).unsqueeze(0)], dim=1)
    
    # Scenario 2: Original intervention (from the sample)
    input_with_intervention = input_combined  # Already have this from earlier
    
    # === Step 10: Make predictions for both scenarios ===
    with torch.no_grad():
        predicted_no_intervention_flat = model(input_no_intervention)
        predicted_with_intervention_flat = predicted_future_flat  # Already computed earlier
    
    # Reshape both predictions
    predicted_no_intervention = predicted_no_intervention_flat.squeeze().numpy().reshape(FUTURE_STEPS, NUM_FEATURES)
    predicted_with_intervention = predicted_future  # Already have this from earlier
    
    # === Step 11: Print comparison ===
    print("Scenario 1: No Intervention (control)")
    print("-" * 80)
    for step in range(FUTURE_STEPS):
        print(f"Step t+{step+1}: {predicted_no_intervention[step]}")
    
    print("\n\nScenario 2: With Intervention")
    print("-" * 80)
    for step in range(FUTURE_STEPS):
        print(f"Step t+{step+1}: {predicted_with_intervention[step]}")
    
    # === Step 12: Compute intervention effect ===
    print("\n\nIntervention Effect (Difference)")
    print("(Positive values: intervention increases the feature; Negative: decreases)")
    print("-" * 80)
    intervention_effect = predicted_with_intervention - predicted_no_intervention
    for step in range(FUTURE_STEPS):
        mean_effect = np.abs(intervention_effect[step]).mean()
        print(f"Step t+{step+1}: {intervention_effect[step]}")
        print(f"          Mean absolute effect: {mean_effect:.4f}")
    
    # Overall intervention strength
    overall_effect = np.abs(intervention_effect).mean()
    print(f"\nOverall Mean Intervention Effect Magnitude: {overall_effect:.4f}")
    print("=" * 80)
    
    # === Step 12b: Intervention Impact Score ===
    # This metric quantifies how much the intervention changes predicted future outcomes
    # Higher score = intervention has a larger effect on system evolution
    intervention_impact_score = np.mean(np.abs(predicted_with_intervention - predicted_no_intervention))
    
    print("\n" + "=" * 80)
    print(f"INTERVENTION IMPACT SCORE: {intervention_impact_score:.4f}")
    print("-" * 80)
    print("Interpretation:")
    print(f"  - The intervention changes the predicted trajectory by an average of {intervention_impact_score:.4f} per feature")
    print(f"  - Higher scores indicate stronger intervention effects")
    print(f"  - This score summarizes the total magnitude of intervention influence")
    print("=" * 80)
    
    # === Step 13: Visualization of intervention effects ===
    print("\nCreating intervention comparison plot...")
    
    # Extract feature values for both scenarios
    no_intervention_values = predicted_no_intervention[:, PLOT_FEATURE_INDEX]
    with_intervention_values = predicted_with_intervention[:, PLOT_FEATURE_INDEX]
    
    # Create the comparison plot
    plt.figure(figsize=(8, 5))
    
    # Plot both trajectories
    plt.plot(time_steps, no_intervention_values, 'g-o', label='No Intervention', linewidth=2, markersize=6)
    plt.plot(time_steps, with_intervention_values, 'm-s', label='With Intervention', linewidth=2, markersize=6)
    
    # Add labels and title
    plt.xlabel('Future Time Steps')
    plt.ylabel(f'Feature {PLOT_FEATURE_INDEX} Value')
    plt.title(f'Intervention Effect on System Trajectory: Feature {PLOT_FEATURE_INDEX}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show the plot
    plt.show(block=False)
    plt.pause(10)  # Display both figures for 10 seconds then continue execution


if __name__ == "__main__":
    main()

