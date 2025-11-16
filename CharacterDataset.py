import torch
from torch.utils.data import Dataset
from logger import setup_logger
from PIL import Image
import numpy as np
import os

logger = setup_logger(__name__, log_file="logs/dataset.log")

class CharacterDataset(Dataset):
    def __init__(self, image_dir: str, resolution=1024, dtype=torch.float16):
        self.image_dir = image_dir
        self.resolution = resolution
        self.dtype = dtype
        self.image_paths = []

        # Find all image files (png, jpg, jpeg)
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        for filename in sorted(os.listdir(image_dir)):
            if os.path.splitext(filename)[1].lower() in valid_extensions:
                self.image_paths.append(os.path.join(image_dir, filename))

        logger.info("Initialized CharacterDataset with %d images (resolution=%d, dtype=%s)",
                    len(self.image_paths), resolution, dtype)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img = img.resize((self.resolution, self.resolution))
            img_array = np.array(img)
            tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 127.5 - 1
            # Cast to target dtype
            tensor = tensor.to(self.dtype)

            # Load corresponding txt file
            base_name = os.path.splitext(image_path)[0]
            txt_path = base_name + ".txt"
            prompt = ""
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    prompt = f.read().strip()
                logger.debug("Loaded prompt from %s: %s", txt_path, prompt)

            return {"pixel_values": tensor, "prompt": prompt}
        except Exception as e:
            logger.error("Error loading image %s: %s", image_path, str(e))
            raise