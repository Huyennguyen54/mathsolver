model.save_pretrained("trained-model")
PEFT_MODEL = "huyennguyen5492/vinallama-peft-7b-math-solver"

model.push_to_hub(
    PEFT_MODEL, use_auth_token=True
)