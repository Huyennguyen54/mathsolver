training_args = transformers.TrainingArguments(
      per_device_train_batch_size=1,
      gradient_accumulation_steps=4,
      num_train_epochs=1,
      learning_rate=2e-4,
      fp16=True,
      save_total_limit=3,
      logging_steps=1,
      output_dir="experiments",
      optim="paged_adamw_8bit",
      lr_scheduler_type="cosine",
      warmup_ratio=0.05,
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=choices_data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()