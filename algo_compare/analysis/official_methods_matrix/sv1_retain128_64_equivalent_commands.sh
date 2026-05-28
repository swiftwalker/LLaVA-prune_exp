#!/usr/bin/env bash
set -euo pipefail

# Generated configuration commands only. Review GPU placement/output dirs before launching.
# Each command uses local unified eval via algo_compare/scripts/run_official.py --eval.

# target=128 dataset=gqa label=SparseVLM-v1-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset gqa --variant sparsevlm_v1 --retain-token 128 --use-version 1_0 --eval

# target=128 dataset=textvqa label=SparseVLM-v1-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset textvqa --variant sparsevlm_v1 --retain-token 128 --use-version 1_0 --eval

# target=128 dataset=pope label=SparseVLM-v1-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset pope --variant sparsevlm_v1 --retain-token 128 --use-version 1_0 --eval

# target=128 dataset=mme label=SparseVLM-v1-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset mme --variant sparsevlm_v1 --retain-token 128 --use-version 1_0 --eval

# target=128 dataset=scienceqa label=SparseVLM-v1-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset scienceqa --variant sparsevlm_v1 --retain-token 128 --use-version 1_0 --eval

# target=128 dataset=gqa label=SparseVLM-v2-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset gqa --variant sparsevlm_v2 --retain-token 128 --use-version 2_0 --eval

# target=128 dataset=textvqa label=SparseVLM-v2-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset textvqa --variant sparsevlm_v2 --retain-token 128 --use-version 2_0 --eval

# target=128 dataset=pope label=SparseVLM-v2-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset pope --variant sparsevlm_v2 --retain-token 128 --use-version 2_0 --eval

# target=128 dataset=mme label=SparseVLM-v2-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset mme --variant sparsevlm_v2 --retain-token 128 --use-version 2_0 --eval

# target=128 dataset=scienceqa label=SparseVLM-v2-128
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset scienceqa --variant sparsevlm_v2 --retain-token 128 --use-version 2_0 --eval

# target=128 dataset=gqa label=FastV-sv1r128compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset gqa --variant fastv_token_mask --fastv-k 2 --fastv-r 0.833333 --fastv-attention-rank 96 --eval

# target=128 dataset=textvqa label=FastV-sv1r128compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset textvqa --variant fastv_token_mask --fastv-k 2 --fastv-r 0.833333 --fastv-attention-rank 96 --eval

# target=128 dataset=pope label=FastV-sv1r128compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset pope --variant fastv_token_mask --fastv-k 2 --fastv-r 0.833333 --fastv-attention-rank 96 --eval

# target=128 dataset=mme label=FastV-sv1r128compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset mme --variant fastv_token_mask --fastv-k 2 --fastv-r 0.833333 --fastv-attention-rank 96 --eval

# target=128 dataset=scienceqa label=FastV-sv1r128compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset scienceqa --variant fastv_token_mask --fastv-k 2 --fastv-r 0.833333 --fastv-attention-rank 96 --eval

# target=128 dataset=gqa label=PDROP-sv1r128compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset gqa --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.526043,0.190974,0.062502]' --eval

# target=128 dataset=textvqa label=PDROP-sv1r128compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset textvqa --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.526043,0.190974,0.062502]' --eval

# target=128 dataset=pope label=PDROP-sv1r128compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset pope --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.526043,0.190974,0.062502]' --eval

# target=128 dataset=mme label=PDROP-sv1r128compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset mme --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.526043,0.190974,0.062502]' --eval

# target=128 dataset=scienceqa label=PDROP-sv1r128compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset scienceqa --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.526043,0.190974,0.062502]' --eval

# target=128 dataset=gqa label=VisionZip-sv1r128compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset gqa --variant visionzip_64 --visionzip-dominant 106 --visionzip-contextual 20 --eval

# target=128 dataset=textvqa label=VisionZip-sv1r128compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset textvqa --variant visionzip_64 --visionzip-dominant 106 --visionzip-contextual 20 --eval

# target=128 dataset=pope label=VisionZip-sv1r128compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset pope --variant visionzip_64 --visionzip-dominant 106 --visionzip-contextual 20 --eval

# target=128 dataset=mme label=VisionZip-sv1r128compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset mme --variant visionzip_64 --visionzip-dominant 106 --visionzip-contextual 20 --eval

# target=128 dataset=scienceqa label=VisionZip-sv1r128compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset scienceqa --variant visionzip_64 --visionzip-dominant 106 --visionzip-contextual 20 --eval

# target=128 dataset=gqa label=DivPrune-sv1r128compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset gqa --variant divprune_r0p098 --divprune-subset-ratio 0.218750 --eval

# target=128 dataset=textvqa label=DivPrune-sv1r128compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset textvqa --variant divprune_r0p098 --divprune-subset-ratio 0.218750 --eval

# target=128 dataset=pope label=DivPrune-sv1r128compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset pope --variant divprune_r0p098 --divprune-subset-ratio 0.218750 --eval

# target=128 dataset=mme label=DivPrune-sv1r128compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset mme --variant divprune_r0p098 --divprune-subset-ratio 0.218750 --eval

# target=128 dataset=scienceqa label=DivPrune-sv1r128compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset scienceqa --variant divprune_r0p098 --divprune-subset-ratio 0.218750 --eval

# target=64 dataset=gqa label=SparseVLM-v1-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset gqa --variant sparsevlm_v1 --retain-token 64 --use-version 1_0 --eval

# target=64 dataset=textvqa label=SparseVLM-v1-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset textvqa --variant sparsevlm_v1 --retain-token 64 --use-version 1_0 --eval

# target=64 dataset=pope label=SparseVLM-v1-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset pope --variant sparsevlm_v1 --retain-token 64 --use-version 1_0 --eval

# target=64 dataset=mme label=SparseVLM-v1-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset mme --variant sparsevlm_v1 --retain-token 64 --use-version 1_0 --eval

# target=64 dataset=scienceqa label=SparseVLM-v1-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset scienceqa --variant sparsevlm_v1 --retain-token 64 --use-version 1_0 --eval

# target=64 dataset=gqa label=SparseVLM-v2-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset gqa --variant sparsevlm_v2 --retain-token 64 --use-version 2_0 --eval

# target=64 dataset=textvqa label=SparseVLM-v2-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset textvqa --variant sparsevlm_v2 --retain-token 64 --use-version 2_0 --eval

# target=64 dataset=pope label=SparseVLM-v2-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset pope --variant sparsevlm_v2 --retain-token 64 --use-version 2_0 --eval

# target=64 dataset=mme label=SparseVLM-v2-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset mme --variant sparsevlm_v2 --retain-token 64 --use-version 2_0 --eval

# target=64 dataset=scienceqa label=SparseVLM-v2-64
python3 algo_compare/scripts/run_official.py --method sparsevlm --dataset scienceqa --variant sparsevlm_v2 --retain-token 64 --use-version 2_0 --eval

# target=64 dataset=gqa label=FastV-sv1r64compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset gqa --variant fastv_token_mask --fastv-k 2 --fastv-r 0.951389 --fastv-attention-rank 28 --eval

# target=64 dataset=textvqa label=FastV-sv1r64compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset textvqa --variant fastv_token_mask --fastv-k 2 --fastv-r 0.951389 --fastv-attention-rank 28 --eval

# target=64 dataset=pope label=FastV-sv1r64compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset pope --variant fastv_token_mask --fastv-k 2 --fastv-r 0.951389 --fastv-attention-rank 28 --eval

# target=64 dataset=mme label=FastV-sv1r64compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset mme --variant fastv_token_mask --fastv-k 2 --fastv-r 0.951389 --fastv-attention-rank 28 --eval

# target=64 dataset=scienceqa label=FastV-sv1r64compute
python3 algo_compare/scripts/run_official.py --method fastv --dataset scienceqa --variant fastv_token_mask --fastv-k 2 --fastv-r 0.951389 --fastv-attention-rank 28 --eval

# target=64 dataset=gqa label=PDROP-sv1r64compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset gqa --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.114585,0.052085,0.029516]' --eval

# target=64 dataset=textvqa label=PDROP-sv1r64compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset textvqa --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.114585,0.052085,0.029516]' --eval

# target=64 dataset=pope label=PDROP-sv1r64compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset pope --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.114585,0.052085,0.029516]' --eval

# target=64 dataset=mme label=PDROP-sv1r64compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset mme --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.114585,0.052085,0.029516]' --eval

# target=64 dataset=scienceqa label=PDROP-sv1r64compute
python3 algo_compare/scripts/run_official.py --method pdrop --dataset scienceqa --variant pdrop_v1_5 --pdrop-layer-list '[2,6,16]' --pdrop-image-token-ratio-list '[0.114585,0.052085,0.029516]' --eval

# target=64 dataset=gqa label=VisionZip-sv1r64compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset gqa --variant visionzip_64 --visionzip-dominant 52 --visionzip-contextual 10 --eval

# target=64 dataset=textvqa label=VisionZip-sv1r64compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset textvqa --variant visionzip_64 --visionzip-dominant 52 --visionzip-contextual 10 --eval

# target=64 dataset=pope label=VisionZip-sv1r64compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset pope --variant visionzip_64 --visionzip-dominant 52 --visionzip-contextual 10 --eval

# target=64 dataset=mme label=VisionZip-sv1r64compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset mme --variant visionzip_64 --visionzip-dominant 52 --visionzip-contextual 10 --eval

# target=64 dataset=scienceqa label=VisionZip-sv1r64compute
python3 algo_compare/scripts/run_official.py --method visionzip --dataset scienceqa --variant visionzip_64 --visionzip-dominant 52 --visionzip-contextual 10 --eval

# target=64 dataset=gqa label=DivPrune-sv1r64compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset gqa --variant divprune_r0p098 --divprune-subset-ratio 0.107639 --eval

# target=64 dataset=textvqa label=DivPrune-sv1r64compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset textvqa --variant divprune_r0p098 --divprune-subset-ratio 0.107639 --eval

# target=64 dataset=pope label=DivPrune-sv1r64compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset pope --variant divprune_r0p098 --divprune-subset-ratio 0.107639 --eval

# target=64 dataset=mme label=DivPrune-sv1r64compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset mme --variant divprune_r0p098 --divprune-subset-ratio 0.107639 --eval

# target=64 dataset=scienceqa label=DivPrune-sv1r64compute
python3 algo_compare/scripts/run_official.py --method divprune --dataset scienceqa --variant divprune_r0p098 --divprune-subset-ratio 0.107639 --eval
