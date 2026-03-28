---
name: run-tests
description: Run the gyaradax test suite
disable-model-invocation: true
argument-hint: [test-filter]
---

Run the pytest suite. Pick a free GPU first.

```bash
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' '$2 ~ /^ *[0-9] MiB/{print $1; exit}')
CUDA_VISIBLE_DEVICES=${FREE_GPU:-0} XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/ -x -q $ARGUMENTS
```

If `$ARGUMENTS` contains a specific test path or `-k` filter, use that. Otherwise run the full suite.
Report the pass/fail summary. If a test fails due to GPU OOM, note it as a pre-existing infra issue.
