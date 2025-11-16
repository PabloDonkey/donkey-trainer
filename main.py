from LoRATrainer import LoRATrainer

trainer = LoRATrainer(
    model_path="/home/pablo/Projects/ComfyUI/models/checkpoints/illustriousXL_v01.safetensors",
    image_dir="/home/pablo/Projects/AI/lora_training/yamaska/dataset",
    output_dir="/home/pablo/Projects/AI/lora_training/yamaska/yamaska_lora"
)

trainer.train(epochs=3)
trainer.save()