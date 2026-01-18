# random masking utils
import torch

def random_masking(x, mask_ratio=0.75):
    """
    Thực hiện random masking trên batch of patch embeddings.
    
    Args:
        x: (B, L, D) - patch embeddings
        mask_ratio: Tỉ lệ patches bị mask (0.75 = 75%)
    
    Returns:
        x_masked: (B, L_visible, D) - chỉ visible patches
        mask: (B, L) - binary mask, 1 = masked, 0 = visible
        ids_restore: (B, L) - indices để khôi phục thứ tự gốc
    """
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))  # Số patches giữ lại (49)
    
    # Random noise để shuffle
    noise = torch.rand(B, L, device=x.device)  # (B, L)
    
    # Sort noise → ids_shuffle chứa thứ tự mới
    ids_shuffle = torch.argsort(noise, dim=1)
    # ids_restore: inverse của ids_shuffle (để unshuffle)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Giữ lại len_keep patches đầu tiên (sau khi shuffle)
    ids_keep = ids_shuffle[:, :len_keep]
    
    # Gather visible patches
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
    
    # Tạo binary mask (1 = masked, 0 = visible)
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    # Unshuffle mask về thứ tự gốc
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore


if __name__ == "__main__":
    x = torch.randn(2, 196, 768)
    x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75)

    assert x_masked.shape == (2, 49, 768), f"Expected (2, 49, 768), got {x_masked.shape}"
    assert mask.shape == (2, 196), f"Expected (2, 196), got {mask.shape}"
    assert mask.sum(dim=1).mean() == 147, f"Expected 147 masked, got {mask.sum(dim=1).mean()}"
    print(f"✓ Masking: {x.shape} → {x_masked.shape}, mask ratio: {mask.sum()/mask.numel():.2%}")