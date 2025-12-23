### vla-getting-started

### Steps
* Step 1: [Collect and create demonstration dataset (.hdf5 file)](step1_dataprep.md)
* Step 2: [Remove no-op and regenerate in higher resolution.](step2_regenerate.md)
* Step 3: [Create rlds file from hdf5 file](step3_rlds.md)
* Step 4: [Create LeRobot file from step 3 generated file](step4_lerobot_format.md)
* Step 5: [Finetune pi0.5](step5_finetune.md)

Note: Step 3,4 can be reduced to one step. Right now it follows OpenVLA strategy.

```
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_libero_low_mem_finetune --policy.dir=checkpoints/pi0_fast_libero_low_mem_finetune/my_experiment90/10000

python -m ensurepip --upgrade
python -m pip install --upgrade pip ipykernel
python -m ipykernel install --user \
  --name myproject-venv \
  --display-name "Python (.venv: myproject)"

libero_inf_1.ipynb
```
