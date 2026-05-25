# Official SparseVLM Notes

Official SparseVLM code is fetched from:

```text
https://github.com/Gumpest/SparseVLMs.git
```

The first pinned commit for this workspace is:

```text
87fe4319430e079a778c25d4ddf4588a3c4038ca
```

The official evaluation shell scripts under `scripts/v1_5/eval/` contain
environment-specific absolute paths, so the wrappers in this directory call the
official Python inference module directly and provide local paths explicitly.

Important reproduction fields to record for every official run:

- `official_repo_commit`
- `USE_VERSION`
- `RETAIN_TOKN`
- model path and model name
- dataset question file and image folder
- output answers file and metric summary

Do not compare local `entropy_exp` absolute scores against paper numbers unless
the official implementation path above was used.
