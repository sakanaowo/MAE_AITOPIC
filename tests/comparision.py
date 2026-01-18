import torch
import sys
import os
sys.path.insert(0, '.')
official_mae_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'official_mae')
sys.path.insert(0, official_mae_path)

# Load official model
import models_mae as official_mae

# Load our implementation
from mae.mae import MAE

def compare_models():
    # Create models
    our_model = MAE()
    official_model = official_mae.mae_vit_base_patch16()
    
    # Compare param counts
    our_params = sum(p.numel() for p in our_model.parameters())
    official_params = sum(p.numel() for p in official_model.parameters())
    
    print(f"Our params: {our_params/1e6:.1f}M")
    print(f"Official params: {official_params/1e6:.1f}M")
    
    # Forward pass comparison
    x = torch.randn(1, 3, 224, 224)
    
    our_loss, _, _ = our_model(x)
    official_loss, _, _ = official_model(x)
    
    print(f"Our loss: {our_loss.item():.4f}")
    print(f"Official loss: {official_loss.item():.4f}")
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    compare_models()