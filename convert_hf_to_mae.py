"""
Convert HuggingFace ViTMAEForPreTraining weights → flat Facebook MAE format.

HuggingFace stores Q, K, V separately; Facebook/custom uses merged qkv.
Output format: {'model': state_dict} compatible with load_pretrained().

Usage:
    python convert_hf_to_mae.py --model facebook/vit-mae-base --output checkpoints/official_checkpoints/mae_visualize_vit_base.pth
    python convert_hf_to_mae.py --model facebook/vit-mae-huge --output checkpoints/official_checkpoints/mae_visualize_vit_huge.pth
"""

import argparse
import torch
import re


def convert_hf_to_facebook_format(hf_state_dict):
    """
    Convert HuggingFace ViTMAEForPreTraining state_dict to flat Facebook MAE format.
    
    Key mapping:
        HuggingFace                                    → Facebook (flat)
        ──────────────────────────────────────────────────────────────────
        vit.embeddings.cls_token                       → cls_token
        vit.embeddings.position_embeddings             → pos_embed
        vit.embeddings.patch_embeddings.projection.*   → patch_embed.proj.*
        vit.layernorm.*                                → norm.*
        vit.encoder.layer.N.layernorm_before.*         → blocks.N.norm1.*
        vit.encoder.layer.N.layernorm_after.*          → blocks.N.norm2.*
        vit.encoder.layer.N.attention.attention.{q,k,v}  → blocks.N.attn.qkv (merged)
        vit.encoder.layer.N.attention.output.dense.*   → blocks.N.attn.proj.*
        vit.encoder.layer.N.intermediate.dense.*       → blocks.N.mlp.fc1.*
        vit.encoder.layer.N.output.dense.*             → blocks.N.mlp.fc2.*
        
        decoder.decoder_embed.*                        → decoder_embed.*
        decoder.mask_token                             → mask_token
        decoder.decoder_pos_embed                      → decoder_pos_embed
        decoder.decoder_norm.*                         → decoder_norm.*
        decoder.decoder_pred.*                         → decoder_pred.*
        decoder.decoder_layers.N.layernorm_before.*    → decoder_blocks.N.norm1.*
        decoder.decoder_layers.N.layernorm_after.*     → decoder_blocks.N.norm2.*
        decoder.decoder_layers.N.attention.attention.{q,k,v}  → decoder_blocks.N.attn.qkv (merged)
        decoder.decoder_layers.N.attention.output.dense.*     → decoder_blocks.N.attn.proj.*
        decoder.decoder_layers.N.intermediate.dense.*  → decoder_blocks.N.mlp.fc1.*
        decoder.decoder_layers.N.output.dense.*        → decoder_blocks.N.mlp.fc2.*
    """
    converted = {}
    
    # Collect Q, K, V for merging (indexed by block prefix)
    qkv_cache = {}
    
    for hf_key, value in hf_state_dict.items():
        fb_key = _map_single_key(hf_key)
        
        if fb_key is None:
            # This is a Q, K, or V key -- needs merging
            _collect_qkv(hf_key, value, qkv_cache)
        else:
            converted[fb_key] = value
    
    # Merge Q, K, V into qkv
    for prefix, parts in qkv_cache.items():
        for suffix in ('weight', 'bias'):
            q = parts[f'query.{suffix}']
            k = parts[f'key.{suffix}']
            v = parts[f'value.{suffix}']
            merged = torch.cat([q, k, v], dim=0)
            converted[f'{prefix}.qkv.{suffix}'] = merged
    
    return converted


def _map_single_key(hf_key):
    """Map a single HF key to Facebook flat key. Returns None if it's a Q/K/V key (needs merging)."""
    
    # === ENCODER ===
    if hf_key == 'vit.embeddings.cls_token':
        return 'cls_token'
    if hf_key == 'vit.embeddings.position_embeddings':
        return 'pos_embed'
    if hf_key.startswith('vit.embeddings.patch_embeddings.projection.'):
        suffix = hf_key.split('vit.embeddings.patch_embeddings.projection.')[1]
        return f'patch_embed.proj.{suffix}'
    if hf_key.startswith('vit.layernorm.'):
        suffix = hf_key.split('vit.layernorm.')[1]
        return f'norm.{suffix}'
    
    # Encoder transformer blocks
    m = re.match(r'vit\.encoder\.layer\.(\d+)\.(.+)', hf_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        return _map_block_key(f'blocks.{idx}', rest)
    
    # === DECODER ===
    if hf_key.startswith('decoder.decoder_embed.'):
        suffix = hf_key.split('decoder.decoder_embed.')[1]
        return f'decoder_embed.{suffix}'
    if hf_key == 'decoder.mask_token':
        return 'mask_token'
    if hf_key == 'decoder.decoder_pos_embed':
        return 'decoder_pos_embed'
    if hf_key.startswith('decoder.decoder_norm.'):
        suffix = hf_key.split('decoder.decoder_norm.')[1]
        return f'decoder_norm.{suffix}'
    if hf_key.startswith('decoder.decoder_pred.'):
        suffix = hf_key.split('decoder.decoder_pred.')[1]
        return f'decoder_pred.{suffix}'
    
    # Decoder transformer blocks
    m = re.match(r'decoder\.decoder_layers\.(\d+)\.(.+)', hf_key)
    if m:
        idx, rest = m.group(1), m.group(2)
        return _map_block_key(f'decoder_blocks.{idx}', rest)
    
    raise ValueError(f'Unknown HuggingFace key: {hf_key}')


def _map_block_key(block_prefix, rest):
    """Map block-level HF key suffix to Facebook format. Returns None for Q/K/V."""
    
    if rest.startswith('layernorm_before.'):
        suffix = rest.split('layernorm_before.')[1]
        return f'{block_prefix}.norm1.{suffix}'
    if rest.startswith('layernorm_after.'):
        suffix = rest.split('layernorm_after.')[1]
        return f'{block_prefix}.norm2.{suffix}'
    
    # Q, K, V → needs merging, return None
    if re.match(r'attention\.attention\.(query|key|value)\.', rest):
        return None
    
    # Attention output projection
    if rest.startswith('attention.output.dense.'):
        suffix = rest.split('attention.output.dense.')[1]
        return f'{block_prefix}.attn.proj.{suffix}'
    
    # MLP
    if rest.startswith('intermediate.dense.'):
        suffix = rest.split('intermediate.dense.')[1]
        return f'{block_prefix}.mlp.fc1.{suffix}'
    if rest.startswith('output.dense.'):
        suffix = rest.split('output.dense.')[1]
        return f'{block_prefix}.mlp.fc2.{suffix}'
    
    raise ValueError(f'Unknown block key: {block_prefix}.{rest}')


def _collect_qkv(hf_key, value, cache):
    """Collect Q, K, V tensors for later merging."""
    # Match encoder: vit.encoder.layer.N.attention.attention.{query|key|value}.{weight|bias}
    m = re.match(r'vit\.encoder\.layer\.(\d+)\.attention\.attention\.(query|key|value)\.(weight|bias)', hf_key)
    if m:
        idx, qkv_type, suffix = m.group(1), m.group(2), m.group(3)
        prefix = f'blocks.{idx}.attn'
        cache.setdefault(prefix, {})[f'{qkv_type}.{suffix}'] = value
        return
    
    # Match decoder: decoder.decoder_layers.N.attention.attention.{query|key|value}.{weight|bias}
    m = re.match(r'decoder\.decoder_layers\.(\d+)\.attention\.attention\.(query|key|value)\.(weight|bias)', hf_key)
    if m:
        idx, qkv_type, suffix = m.group(1), m.group(2), m.group(3)
        prefix = f'decoder_blocks.{idx}.attn'
        cache.setdefault(prefix, {})[f'{qkv_type}.{suffix}'] = value
        return
    
    raise ValueError(f'Expected Q/K/V key, got: {hf_key}')


def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace ViTMAE → Facebook MAE format')
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model ID, e.g. facebook/vit-mae-base')
    parser.add_argument('--output', type=str, required=True,
                        help='Output .pth path')
    args = parser.parse_args()
    
    print(f'Loading HuggingFace model: {args.model}')
    from transformers import ViTMAEForPreTraining
    hf_model = ViTMAEForPreTraining.from_pretrained(args.model)
    hf_sd = hf_model.state_dict()
    print(f'  HuggingFace keys: {len(hf_sd)}')
    
    print('Converting to Facebook MAE format...')
    fb_sd = convert_hf_to_facebook_format(hf_sd)
    print(f'  Facebook keys: {len(fb_sd)}')
    
    # Validate: count encoder vs decoder keys
    enc_keys = [k for k in fb_sd if not k.startswith('decoder_') and k != 'mask_token']
    dec_keys = [k for k in fb_sd if k.startswith('decoder_') or k == 'mask_token']
    print(f'  Encoder keys: {len(enc_keys)}, Decoder keys: {len(dec_keys)}')
    
    # Save in Facebook format: {'model': state_dict}
    torch.save({'model': fb_sd}, args.output)
    
    import os
    size_mb = os.path.getsize(args.output) / 1e6
    print(f'Saved: {args.output} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
