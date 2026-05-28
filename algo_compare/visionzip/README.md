# VisionZip Official Wrapper

This directory contains the local wrapper for the official VisionZip method.
Official source is fetched into `third_party/VisionZip` and is not committed.

## Fetch

```bash
bash algo_compare/visionzip/scripts/fetch_official.sh
```

## Dry Run

```bash
python3 algo_compare/scripts/run_official.py \
  --method visionzip \
  --dataset mme \
  --variant visionzip_64 \
  --eval \
  --dry-run
```

## Default Config

- official repo: `https://github.com/dvlab-research/VisionZip`
- variant: `visionzip_64`
- dominant tokens: `54`
- contextual tokens: `10`
- retained visual tokens: `64`

The runner loads local LLaVA first, then applies the official:

```python
from visionzip import visionzip
model = visionzip(model, dominant=54, contextual=10)
```
