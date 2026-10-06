# CHARM — source

| Item | Value |
|---|---|
| Paper | Frasca, Bar-Shalom, Ziser, Maron. *Neural Message-Passing on Attention Graphs for Hallucination Detection*. ICLR 2026 |
| Paper version | arXiv 2509.24770; OpenReview `4twbqwV4br` |
| Code | https://github.com/Noired/charm (owner: Fabrizio Frasca, first author) |
| Commit | `7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61` (2026-07-27) |
| Licence | MIT (LICENSE file and GitHub API). Third-party data terms: PROVENANCE.md upstream. |
| Status | **Adapted**: the adapter calls the official functions. Our earlier reimplementation (stage 6) was removed on 2026-10-06. |

## How to get the code

```bash
git clone https://github.com/Noired/charm original-repos/charm
git -C original-repos/charm checkout 7ab3ab7f7a49fd6ca23defc44ddcd6c794598c61
```

## Upstream dependency stack

The experiments environment (README) uses torch 2.7.0, `torch_geometric` (unpinned),
scikit-learn 1.6.1 and transformers 4.52.4. The upstream transforms define `__call__` and
no `forward`, which works with `torch_geometric` 2.6.1 and fails from 2.7.0 on, where
`BaseTransform.forward` is abstract. This project therefore pins
`torch-geometric==2.6.1` (pyproject.toml). It changes nothing else in the stack: torch
stays 2.13.0. `wandb` is stubbed: it is imported at module level but used only for logging,
which the adapter never enables.
