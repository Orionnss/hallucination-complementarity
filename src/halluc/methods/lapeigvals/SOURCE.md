# LapEigvals — source

| Item | Value |
|---|---|
| Paper | Binkowski et al., *Hallucination Detection in LLMs Using Spectral Features of Attention Maps*, EMNLP 2025 |
| Paper version | arXiv 2502.17598v2 |
| Code | https://github.com/graphml-lab-pwr/lapeigvals |
| Commit | `74f885c69399ed525f4eecd2574c218fac7235d7` (2025-10-18, "Initial commit") |
| Licence | **None.** The repository has no LICENSE file, and the GitHub API reports no licence (checked 2026-10-05). |
| Status | **Adapted**: the adapter calls the official functions. |

## Consequence of the missing licence

Without a licence, we do not have permission to copy or redistribute the code. Thus:

- The code is **not** in this repository. Clone it to `original-repos/lapeigvals` and check
  out the commit above. `.gitignore` excludes `original-repos/`.
- The adapter imports the official modules from that checkout. It does not copy them
  (`src/halluc/methods/upstream.py`).
- Before each import, the loader checks the commit and the sha256 of each file that it
  uses (`upstream_spec.py`). If a file is different, the loader stops.

Before publication, ask the authors for a licence, or for permission.

## How to get the code

```bash
git clone https://github.com/graphml-lab-pwr/lapeigvals original-repos/lapeigvals
git -C original-repos/lapeigvals checkout 74f885c69399ed525f4eecd2574c218fac7235d7
```

## Upstream dependency stack

The upstream `pyproject.toml` pins torch 2.6.0, transformers 4.47.0 and scikit-learn
1.6.1. This project uses torch 2.13.0, transformers 5.15.1 and scikit-learn 1.9.0. We do
not install the upstream package. We import only the five modules in `upstream_spec.py`.
One patch (P1 in METHOD_CARD.md) is necessary because of the scikit-learn version.
