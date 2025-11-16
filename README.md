# donkey-trainer
A lightweight LoRA trainer for character-focused images using a Stable Diffusion UNet adapter.

Train LoRA adapters on a small character image dataset to capture character-specific appearance while keeping the base model frozen. Includes a simple dataset loader, LoRA setup for the UNet, training loop, and save/export utilities.

# Key features
* Simple dataset loader and preprocessing for character images.
* LoRA integration for efficient fine-tuning of the UNet.
* Configurable training loop with device detection and checkpoint/save.
* Optional trigger word support for prompt-based workflows (metadata only).

# Quick usage example
```python
from LoRATrainer import LoRATrainer

trainer = LoRATrainer(
    model_path=`/path/to/model.safetensors`,
    image_dir=`/path/to/images`,
    output_dir=`./my_character_lora`,
    batch_size=4,
    lr=1e-4
)
trainer.train(epochs=10, log_interval=10)
trainer.save()
```
