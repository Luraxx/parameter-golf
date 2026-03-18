# Depth Recurrence: 3 Unique Layers × 5 Repetitions at 768-dim

## Summary

This submission applies **depth recurrence** (layer weight sharing) to aggressively reduce parameter count while increasing effective depth and width. Instead of 9 unique transformer layers consuming 9× the parameter budget, we use **3 unique layers repeated cyclically 5 times = 15 effective layers**.

The freed parameter budget is reinvested in a **wider model (768-dim instead of 512-dim)**, resulting in:
- **More depth**: 15 effective layers vs 9 → better sequential composition
- **More width**: 768 vs 512 dim → richer representations per layer
- **Fewer unique parameters**: ~12.6M vs ~17M → well under 16MB compressed

## Architecture Details

### Depth Recurrence
- **3 unique transformer blocks** (containing Q/K/V/O projections, MLP, RMSNorm, RoPE)
- Each block is reused **5 times** cyclically across 15 effective layer positions
- Block `i` at effective position `j` is `blocks[j % 3]`

### Per-Effective-Layer Specialization
Each of the 15 effective layers has its own lightweight adaptive parameters:
- `eff_attn_scales[i]` — per-dim attention output scaling (768 floats)
- `eff_mlp_scales[i]` — per-dim MLP output scaling (768 floats)
- `eff_resid_mixes[i]` — 2×768 residual/skip-connection mixing weights

These cost only ~4×768 = 3072 parameters per effective layer (46K total), but allow each repetition of a shared block to behave differently.

### U-Net Skip Connections
The U-Net skip structure from the baseline is preserved, operating on effective layer indices:
- Encoder half (layers 0–6): accumulates skip tensors
- Decoder half (layers 7–14): consumes reversed skips with learned `skip_weights`

### Other Settings
- Vocab size: 1024 (SentencePiece BPE)
- Sequence length: 1024
- Tied embeddings
- 12 attention heads, 4 KV heads (GQA)
- 2× MLP expansion (relu²)
- Logit softcap: 30.0

## Parameter Budget

| Component | Parameters |
|-----------|-----------|
| Token embedding (1024 × 768) | 786,432 |
| 3 unique blocks (Q+K+V+O + MLP) | ~11,665,920 |
| Per-effective-layer scales (15 × ~3K) | ~46,080 |
| Skip weights (7 × 768) | 5,376 |
| Other (q_gain, etc.) | ~130K |
| **Total** | **~12.6M** |

## Training

Optimizer setup matches baseline:
- Muon for matrix params (shared block weights)
- Adam for embeddings, scalar/control params, and per-effective-layer scales
- Warmdown with wallclock-aware scheduling

## Command

```bash
NCCL_IB_DISABLE=1 \
RUN_ID=depth_recurrence_3x5_768 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=15 \
NUM_UNIQUE_LAYERS=3 \
MODEL_DIM=768 \
NUM_HEADS=12 \
NUM_KV_HEADS=4 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metric

- Final `val_bpb`: TBD (requires 8×H100 run)
- Estimated compressed size: ~10–12MB (well under 16MB)
