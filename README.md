# chip404-solver

CUDA PoW solver for the Chip404 (CHIP) free mint lane on BSC.
Finds a nonce with `keccak256(keccak256("chip404.trial.v1") || wallet || nonce) <= targetNow()`
for wallet `0x1f0aF4Dcd33Bbe88B62D5ad6307cfE2E9aCe13Ea` (contract
`0xAB6b18C626daaDd1ff7Cc45F402141153a5a9759`).

## Kaggle notebook (GPU T4 x2, Internet ON)

```python
!git clone -q https://github.com/iyuss0401/chip404-solver.git || (cd chip404-solver && git pull -q)
%cd chip404-solver
!python3 run.py
```

Then **Save Version -> Save & Run All** so the search runs server-side to
completion. The driver polls the chain itself, verifies any candidate nonce
locally against the fresh target, and writes `result.json` into
`/kaggle/working` the moment a usable nonce is found (status `margin`,
`plain-sprint`, or `plain-at-cap`).

- `solver.cu` - CUDA keccak search, one shard per GPU, dual threshold
  (plain target + target>>2 margin), best-hash tracking.
- `run.py` - driver: chain reads, compile, shard orchestration, sprint
  rules near epoch boundaries, local verification.
