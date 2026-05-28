# VisionZip Official Repo Notes

- Repository: `https://github.com/dvlab-research/VisionZip`
- Target commit: `8f86b55c6f000eb033e6912538af2dd7dcb30502`
- Main LLaVA entrypoint: install/import package `visionzip`, then call
  `visionzip(model, dominant=54, contextual=10)` after loading LLaVA.
- The official repository provides a lightweight package rather than a full
  standalone LLaVA evaluation checkout, so the local wrapper uses this
  workspace's LLaVA implementation and injects the official VisionZip patch.
