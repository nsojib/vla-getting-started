    TrainConfig(
        name="pi05_microwave_open_libero_like_low_mem_finetune",
        model=pi0_config.Pi0Config(
            pi05=True, 
            action_horizon=10, 
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="nsojib/microwave_open_libero_like",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
         
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,

        # LoRA: freeze non-LoRA weights (must match the model config above)
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=10,
        ).get_freeze_filter(),

        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
         
    ),
