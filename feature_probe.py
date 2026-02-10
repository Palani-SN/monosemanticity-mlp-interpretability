import torch
import torch.nn as nn
from mlp_definition import InterpretabilityMLP
from sae_definition import SparseAutoencoder

def inspect_features(mlp_path, sae_path, test_input, exp_output):
    # 1. Setup Models
    mlp = InterpretabilityMLP()
    mlp.load_state_dict(torch.load(mlp_path))
    mlp.eval()
    
    # Enable activations capture
    mlp.activations = {} 
    
    sae = SparseAutoencoder(input_dim=512, dict_size=2048)
    sae.load_state_dict(torch.load(sae_path))
    sae.eval()

    # 2. Process Input through MLP
    # Convert list to tensor and add batch dimension
    input_tensor = torch.tensor([test_input], dtype=torch.float32)
    with torch.no_grad():
        output = mlp(input_tensor)
        mlp_acts = mlp.activations['layer2']

    # 3. Process MLP Activations through SAE
    with torch.no_grad():
        _, sae_hidden = sae(mlp_acts)

    # 4. Find Active Features
    # Get indices of features that are non-zero
    active_indices = torch.where(sae_hidden[0] > 0.01)[0].tolist()
    values = sae_hidden[0][active_indices].tolist()
    
    print(f"\n--- Interpretability Report ---")
    print(f"Sample Input: {test_input}     |     Expected Output: {exp_output}")
    print(f"MLP Output: {output.item():.4f}")
    print(f"Number of active SAE features: {len(active_indices)}")
    
    # Sort by activation strength
    sorted_features = sorted(zip(active_indices, values), key=lambda x: x[1], reverse=True)
    
    print("\nTop Active Features (Monosemantic Candidates):")
    for idx, val in sorted_features[:5]:
        print(f"Feature #{idx:4} | Activation: {val:.4f}")

if __name__ == "__main__":

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 1, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 1. Index is 3.
        2, 9, 4, 7, 1   # Col 2: value at 1 is 9. Index is 1.
    ]
    # Expected math: abs(9 - 1) = 8.0
    inspect_features("perfect_mlp.pth", "sae_model.pth", sample_input, exp_output=8.0)

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 2, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 2. Index is 3.
        2, 8, 4, 7, 1   # Col 2: value at 1 is 8. Index is 1.
    ]
    # Expected math: abs(2 - 8) = 6.0
    inspect_features("perfect_mlp.pth", "sae_model.pth", sample_input, exp_output=6.0)

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 3, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 3. Index is 3.
        2, 7, 4, 7, 1   # Col 2: value at 1 is 7. Index is 1.
    ]
    # Expected math: abs(3 - 7) = 4.0
    inspect_features("perfect_mlp.pth", "sae_model.pth", sample_input, exp_output=4.0)

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 4, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 4. Index is 3.
        2, 5, 4, 7, 1   # Col 2: value at 1 is 5. Index is 1.
    ]
    # Expected math: abs(4 - 5) = 1.0
    inspect_features("perfect_mlp.pth", "sae_model.pth", sample_input, exp_output=1.0)

    # --- Test Case ---
    # Recall your logic: abs( inp[0][index1] - inp[1][index2] )
    # Let's create an input where index1 (last element of col 1) is 3
    # and index2 (last element of col 2) is 1.
    sample_input = [
        8, 9, 5, 5, 3,  # Col 1: values at 0,1,2,3 are 8, 9, 5, 5. Index is 3.
        2, 4, 4, 7, 1   # Col 2: value at 1 is 4. Index is 1.
    ]
    # Expected math: abs(5 - 4) = 1.0
    inspect_features("perfect_mlp.pth", "sae_model.pth", sample_input, exp_output=1.0)