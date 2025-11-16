import torch
from diffusers import StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model

from logger import setup_logger

logger = setup_logger(__name__, log_file="logs/model_loader.log")

class ModelLoader:
    @staticmethod
    def load_model(model_path, dtype=torch.float16,use_safetensors: bool = True):
        logger.info("Loading model from %s", model_path)
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path, torch_dtype=dtype, use_safetensors=use_safetensors
            )
            logger.info("Model loaded")
            return pipe
        except Exception as e:
            logger.exception("Failed to load model: %s", e)
            raise

    @staticmethod
    def setup_lora(unet, r: int = 16, lora_alpha: int = 32, target_modules=None, lora_dropout: float = 0.05):
        target_modules = target_modules or ["to_k", "to_v", "to_q", "to_out.0"]
        logger.info("Configuring LoRA (r=%d, alpha=%d)", r, lora_alpha)
        lora_config = LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules, lora_dropout=lora_dropout)
        unet = get_peft_model(unet, lora_config)
        logger.info("LoRA configured")
        return unet