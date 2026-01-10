### vla-getting-started

Download vla ready rlds dataset for the two task from libero 90 [download](https://universitysystemnh-my.sharepoint.com/:u:/g/personal/ns1254_usnh_edu/IQCQMQd5R5IUTYfRT-alBUBSAfh_hGrBalfUbrq6UFxKnmY?download=1)

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

<b>Openvla lora finetunning </b>

Around 6.30hrs on a A40
```
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir /home/ns1254/tensorflow_datasets \
  --dataset_name libero_90_no_noops \
  --run_root_dir /home/ns1254/openvla/experiments/out_libero2 \
  --adapter_tmp_dir /home/ns1254/openvla/experiments/tmp2 \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project vla \
  --wandb_entity openvla \
  --save_steps 2000 \
  --max_steps 8000
```

<b>Inference </b>
```
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint experiments/out_libero2/openvla-7b+libero_90_no_noops+b16+lr-0.0005+lora-r32+dropout-0.0--image_aug \
  --task_suite_name libero_90 \
  --center_crop True
```
