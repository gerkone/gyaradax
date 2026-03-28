---
name: run-sim
description: Run a gyaradax simulation from a YAML config
disable-model-invocation: true
argument-hint: <config.yaml> [--kinetic] [--from-scratch] [--device=N]
---

Run a gyaradax simulation. Arguments are passed directly to `scripts/run.py`.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=. python -u scripts/run.py $ARGUMENTS
```

If `--device` is not specified, check GPU availability with `nvidia-smi` and pick a free one.
If `--kinetic` is passed, adaptive CFL is automatically enabled.
For long runs, wrap in `nohup` and redirect to a log file.
