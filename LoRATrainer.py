import torch
from torch.utils.data import DataLoader

from ModelLoader import ModelLoader
from CharacterDataset import CharacterDataset
from logger import setup_logger

logger = setup_logger(__name__, log_file="logs/trainer.log")

class LoRATrainer:

    def __init__(self, model_path: str, image_dir: str, output_dir: str = "", batch_size: int = 1, lr: float = 1e-4, dtype=torch.float16):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)
        self.pipe = ModelLoader.load_model(model_path, dtype=dtype)
        self.unet = ModelLoader.setup_lora(self.pipe.unet)
        self.unet.to(self.device)
        self.text_encoder = self.pipe.text_encoder.to(self.device)
        self.text_encoder_2 = self.pipe.text_encoder_2.to(self.device)
        self.tokenizer = self.pipe.tokenizer
        self.tokenizer_2 = self.pipe.tokenizer_2
        self.dataset = CharacterDataset(image_dir, dtype=dtype)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        self.output_dir = output_dir if output_dir else "lora_model"
        logger.info("Trainer initialized (output=%s, batch_size=%d, lr=%g, dtype=%s)", self.output_dir, batch_size, lr, dtype)

    def train(self, epochs: int = 10, log_interval: int = 1):
        self.unet.train()
        self.pipe.vae.eval()
        self.pipe.vae.to(self.device)
        self.text_encoder.eval()
        self.text_encoder_2.eval()

        for epoch in range(epochs):
            logger.info("Starting epoch %d/%d", epoch + 1, epochs)
            for i, batch in enumerate(self.dataloader, 1):
                try:
                    # FIX 1: Validate batch contents
                    if batch is None or batch.get("pixel_values") is None:
                        logger.error("Batch is None or missing pixel_values at epoch %d batch %d",
                                   epoch + 1, i)
                        continue

                    pixel_values = batch["pixel_values"].to(self.device)
                    prompts = batch.get("prompt")
                    batch_size = pixel_values.shape[0]

                    # FIX 2: Validate prompts
                    if prompts is None:
                        logger.error("Prompts are None at epoch %d batch %d", epoch + 1, i)
                        continue

                    if isinstance(prompts, str):
                        prompts_list = [prompts] * batch_size
                    elif isinstance(prompts, (list, tuple)):
                        prompts_list = list(prompts)
                    else:
                        logger.error("Unexpected prompts type: %s at epoch %d batch %d",
                                   type(prompts), epoch + 1, i)
                        continue

                    if any(p is None for p in prompts_list):
                        logger.error("Some prompts are None at epoch %d batch %d", epoch + 1, i)
                        continue

                    with torch.no_grad():
                        # FIX 3: Validate prompt embeddings
                        prompt_embeds = self.pipe.encode_prompt(
                            prompt=prompts_list,
                            device=self.device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=[None] * batch_size
                        )

                        if prompt_embeds is None:
                            logger.error("prompt_embeds is None at epoch %d batch %d", epoch + 1, i)
                            continue

                        if prompt_embeds[0] is None or prompt_embeds[1] is None:
                            logger.error("Prompt embedding tensors are None at epoch %d batch %d",
                                       epoch + 1, i)
                            continue

                        encoder_hidden_states = prompt_embeds[0]
                        text_embeds = prompt_embeds[1]

                        # FIX 4: Validate VAE output
                        vae_output = self.pipe.vae.encode(pixel_values)

                        if vae_output is None:
                            logger.error("VAE output is None at epoch %d batch %d", epoch + 1, i)
                            continue

                        latents = vae_output.latent_dist.sample()

                        if latents is None:
                            logger.error("Latents are None after VAE encoding at epoch %d batch %d",
                                       epoch + 1, i)
                            continue

                        latents = latents * 0.13025

                    timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)

                    # SDXL requires time_ids (original resolution and crop top-left)
                    original_size = (1024, 1024)
                    crops_coords_top_left = (0, 0)
                    target_size = (1024, 1024)

                    add_time_ids = torch.tensor(
                        [original_size + crops_coords_top_left + target_size],
                        dtype=self.pipe.dtype,
                        device=self.device
                    ).repeat(batch_size, 1)

                    # FIX 5: Validate UNet output
                    unet_output = self.unet(
                        latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs={"text_embeds": text_embeds, "time_ids": add_time_ids}
                    )

                    if unet_output is None:
                        logger.error("UNet output is None at epoch %d batch %d", epoch + 1, i)
                        continue

                    noise_pred = unet_output.sample

                    if noise_pred is None:
                        logger.error("noise_pred is None at epoch %d batch %d", epoch + 1, i)
                        continue

                    loss = noise_pred.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if i % log_interval == 0:
                        logger.info("Epoch %d, Batch %d: Loss = %.4f", epoch + 1, i, loss.item())
                except Exception as e:
                    logger.exception("Error during training at epoch %d batch %d: %s", epoch + 1, i, e)
                    raise
            logger.info("Finished epoch %d", epoch + 1)

    def save(self):
        logger.info("Saving LoRA to %s", self.output_dir)
        try:
            self.unet.save_pretrained(self.output_dir)
            logger.info("Saved LoRA successfully")
        except Exception as e:
            logger.exception("Failed to save LoRA: %s", e)
            raise