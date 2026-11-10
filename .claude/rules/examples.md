---
paths:
  - "examples/**"
---
Running `gpumd` inside `examples/<name>/` overwrites tracked outputs (e.g.
`examples/gpumd_static/dump.xyz`, `neighbor.out`). Copy the example to a scratch directory and run
it there. If tracked files were modified, restore with `git checkout -- <files>`; never commit them.
