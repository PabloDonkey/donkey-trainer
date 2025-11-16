import torch
from torch.utils.data import DataLoader

from ModelLoader import ModelLoader
from CharacterDataset import CharacterDataset
from logger import setup_logger

logger = setup_logger(__name__, log_file="logs/trainer.log")

class LoRATrainer:

    def __init__(self, model_path: str, image_dir: str, output_dir: str="./my_character_lora",batch_size: int = 1, lr: float = 1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)
        self.pipe = ModelLoader.load_model(model_path)
        self.unet = ModelLoader.setup_lora(self.pipe.unet)
        self.unet.to(self.device)
        self.dataset = CharacterDataset(image_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        self.output_dir = output_dir
        logger.info("Trainer initialized (output=%s, batch_size=%d, lr=%g)", output_dir, batch_size, lr)

    def train(self, epochs: int = 10, log_interval: int = 1):
        self.unet.train()
        self.pipe.vae.eval()
        for epoch in range(epochs):
            logger.info("Starting epoch %d/%d", epoch, epochs)
            for i,batch in enumerate(self.dataloader, 1):
                try:
                    with torch.no_grad():
                        latents = self.pipe.vae.encode(batch["pixel_values"].to(self.device)).latent_dist.sample()
                    loss = self.unet(latents, timestep=torch.randint(0, 1000, (1,)).to(self.device)).sample.mean()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if batch[0] % log_interval == 0:
                        logger.info("Epoch %d, Batch %d: Loss = %.4f", epoch + 1, batch[0], loss.item())
                except Exception as e:
                    logger.exception("Error during training at epoch %d batch %d: %s", epoch, i, e)
                    raise
            logger.info("Finished epoch %d", epoch)

    def save(self):
        logger.info("Saving LoRA to %s", self.output_dir)
        try:
            self.unet.save_pretrained(self.output_dir)
            logger.info("Saved LoRA successfully")
        except Exception as e:
            logger.exception("Failed to save LoRA: %s", e)
            raise