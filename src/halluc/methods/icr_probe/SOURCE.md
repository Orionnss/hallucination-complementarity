# ICR Probe — source

| Item | Value |
|---|---|
| Paper | Zhang, Hu, Zhang, Zhang, Wan. *ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs*. ACL 2025 (Long), pages 17986–18002 |
| Paper version | arXiv 2507.16488 |
| Code | https://github.com/XavierZhang2002/ICR_Probe (owner: Zhenliang Zhang, first author) |
| Commit | `40ec490e762cadbac6bcefdc24a8f0d5974e8448` (2026-03-23) |
| Licence | Apache-2.0 (LICENSE file and GitHub API). The README badge says MIT; the LICENSE file is the authority. |
| Status | **Adapted**: the adapter calls the official functions. |

## How to get the code

```bash
git clone https://github.com/XavierZhang2002/ICR_Probe original-repos/icr_probe
git -C original-repos/icr_probe checkout 40ec490e762cadbac6bcefdc24a8f0d5974e8448
```

The licence permits a copy, but the code stays in `original-repos/` like the other
upstreams, so that one loader verifies every upstream the same way.

## Upstream dependency stack

The repository pins no versions. It needs torch, numpy and scikit-learn, which this
project has. One patch is necessary (P1 in METHOD_CARD.md), for CPU devices only.
