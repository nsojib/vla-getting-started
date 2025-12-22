```
pi0_fast_libero_low_mem_finetune_b
uv run scripts/compute_norm_stats.py --config-name pi0_fast_libero_low_mem_finetune_b
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_libero_low_mem_finetune_b --exp-name=my_experiment_b --overwrite
```

