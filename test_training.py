"""
Test suite to replicate and debug the 'NoneType' object has no attribute 'shape' error
"""
import os
import sys
import torch
import tempfile
import shutil
from PIL import Image
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from CharacterDataset import CharacterDataset
from LoRATrainer import LoRATrainer
from ModelLoader import ModelLoader
from logger import setup_logger

logger = setup_logger(__name__, log_file="logs/test.log")


class TestDataGenerator:
    """Generate test data for debugging"""
    
    @staticmethod
    def create_test_dataset(num_images=2, img_dir=None):
        """Create a temporary directory with test images and captions"""
        if img_dir is None:
            img_dir = tempfile.mkdtemp(prefix="test_dataset_")
        
        os.makedirs(img_dir, exist_ok=True)
        
        for i in range(num_images):
            # Create a dummy image
            img = Image.new('RGB', (256, 256), color=(randint := __import__('random').randint(0, 255), 
                                                       __import__('random').randint(0, 255), 
                                                       __import__('random').randint(0, 255)))
            img_path = os.path.join(img_dir, f"{i+1}.png")
            img.save(img_path)
            
            # Create corresponding caption
            caption_path = os.path.join(img_dir, f"{i+1}.txt")
            with open(caption_path, 'w') as f:
                f.write(f"A test character number {i+1}")
        
        logger.info(f"Created test dataset with {num_images} images in {img_dir}")
        return img_dir


class TestCharacterDataset:
    """Test CharacterDataset for potential None returns"""
    
    @staticmethod
    def test_dataset_loading():
        """Test if dataset properly returns batches without None values"""
        print("\n=== Testing CharacterDataset ===")
        
        test_dir = TestDataGenerator.create_test_dataset(num_images=2)
        
        try:
            dataset = CharacterDataset(test_dir, resolution=256, dtype=torch.float32)
            print(f"✓ Dataset created with {len(dataset)} images")
            
            # Test single item retrieval
            for idx in range(len(dataset)):
                item = dataset[idx]
                assert item is not None, f"Item {idx} is None!"
                assert "pixel_values" in item, f"Item {idx} missing pixel_values!"
                assert "prompt" in item, f"Item {idx} missing prompt!"
                assert item["pixel_values"] is not None, f"pixel_values at index {idx} is None!"
                assert item["prompt"] is not None, f"prompt at index {idx} is None!"
                print(f"✓ Item {idx}: pixel_values shape={item['pixel_values'].shape}, prompt='{item['prompt']}'")
                
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


class TestPromptEncoding:
    """Test prompt encoding pipeline"""
    
    @staticmethod
    def test_encode_prompt_mock():
        """Test if encode_prompt can return None"""
        print("\n=== Testing Prompt Encoding ===")
        
        # Create mock pipeline
        mock_pipe = MagicMock()
        
        # Test 1: Normal case - should return tuple of (encoder_hidden_states, text_embeds)
        print("\nTest 1: Normal encode_prompt")
        mock_pipe.encode_prompt.return_value = (
            torch.randn(1, 77, 768),  # encoder_hidden_states
            torch.randn(1, 1280)      # text_embeds
        )
        result = mock_pipe.encode_prompt(prompt=["test"], device="cpu", num_images_per_prompt=1)
        if result is None:
            print("✗ encode_prompt returned None!")
        else:
            print(f"✓ encode_prompt returned tuple with shapes: {result[0].shape}, {result[1].shape}")
        
        # Test 2: Buggy case - returning None
        print("\nTest 2: encode_prompt returns None")
        mock_pipe.encode_prompt.return_value = None
        result = mock_pipe.encode_prompt(prompt=["test"], device="cpu", num_images_per_prompt=1)
        if result is None:
            print("✗ encode_prompt returned None - THIS WOULD CAUSE THE ERROR!")
            print("  Error would occur when trying: encoder_hidden_states = result[0]")
        
        # Test 3: Buggy case - returning incomplete tuple
        print("\nTest 3: encode_prompt returns incomplete tuple")
        mock_pipe.encode_prompt.return_value = (None, None)
        result = mock_pipe.encode_prompt(prompt=["test"], device="cpu", num_images_per_prompt=1)
        if result[0] is None:
            print("✗ encode_prompt returned None tensors - THIS WOULD CAUSE THE ERROR!")
            print("  Error would occur when trying: latents = self.pipe.vae.encode(encoder_hidden_states)")


class TestVAEEncoding:
    """Test VAE encoding pipeline"""
    
    @staticmethod
    def test_vae_encode_mock():
        """Test if VAE encoding can fail silently"""
        print("\n=== Testing VAE Encoding ===")
        
        # Create mock VAE
        mock_vae = MagicMock()
        
        # Test 1: Normal case
        print("\nTest 1: Normal VAE encoding")
        mock_latent_dist = MagicMock()
        mock_latent_dist.sample.return_value = torch.randn(1, 4, 128, 128)
        mock_vae.encode.return_value = mock_latent_dist
        
        pixel_values = torch.randn(1, 3, 1024, 1024)
        result = mock_vae.encode(pixel_values).latent_dist.sample()
        if result is None:
            print("✗ VAE encoding returned None!")
        else:
            print(f"✓ VAE encoding returned tensor with shape: {result.shape}")
        
        # Test 2: Buggy case - encode returns None
        print("\nTest 2: VAE encode returns None")
        mock_vae.encode.return_value = None
        try:
            result = mock_vae.encode(pixel_values).latent_dist.sample()
            print("✗ VAE returned None - Error: 'NoneType' object has no attribute 'latent_dist'")
        except AttributeError as e:
            print(f"✗ AttributeError caught: {e}")


class TestUNetForward:
    """Test UNet forward pass"""
    
    @staticmethod
    def test_unet_forward_mock():
        """Test if UNet forward can fail"""
        print("\n=== Testing UNet Forward Pass ===")
        
        # Create mock UNet
        mock_unet = MagicMock()
        
        # Test 1: Normal case
        print("\nTest 1: Normal UNet forward")
        mock_output = MagicMock()
        mock_output.sample = torch.randn(1, 4, 128, 128)
        mock_unet.return_value = mock_output
        
        latents = torch.randn(1, 4, 128, 128)
        timesteps = torch.tensor([500])
        encoder_hidden_states = torch.randn(1, 77, 768)
        text_embeds = torch.randn(1, 1280)
        time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]])
        
        result = mock_unet(
            latents, timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}
        ).sample
        
        if result is None:
            print("✗ UNet forward returned None!")
        else:
            print(f"✓ UNet forward returned tensor with shape: {result.shape}")
        
        # Test 2: Buggy case - returns None
        print("\nTest 2: UNet forward returns None")
        mock_unet.return_value = None
        try:
            result = mock_unet(
                latents, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids}
            ).sample
            print(f"✗ UNet returned None - Error: 'NoneType' object has no attribute 'sample'")
        except AttributeError as e:
            print(f"✗ AttributeError caught: {e}")


class TestBatchProcessing:
    """Test batch processing for None values"""
    
    @staticmethod
    def test_batch_with_none_prompts():
        """Test handling of None prompts in batch"""
        print("\n=== Testing Batch Processing ===")
        
        # Create test batch with potential None values
        print("\nTest 1: Batch with valid prompts")
        batch = {
            "pixel_values": torch.randn(2, 3, 256, 256),
            "prompt": ["prompt 1", "prompt 2"]
        }
        
        if batch["prompt"] is None or len(batch["prompt"]) == 0:
            print("✗ Batch prompts are None or empty!")
        else:
            print(f"✓ Batch prompts valid: {batch['prompt']}")
        
        # Test with None prompt
        print("\nTest 2: Batch with None prompts")
        batch = {
            "pixel_values": torch.randn(2, 3, 256, 256),
            "prompt": None
        }
        
        if batch["prompt"] is None:
            print("✗ Batch prompt is None - THIS WOULD CAUSE ENCODE_PROMPT TO FAIL!")
        else:
            print(f"✓ Batch prompt valid: {batch['prompt']}")
        
        # Test with empty string prompts
        print("\nTest 3: Batch with empty string prompts")
        batch = {
            "pixel_values": torch.randn(2, 3, 256, 256),
            "prompt": ["", ""]
        }
        
        if batch["prompt"] is None or all(p == "" for p in batch["prompt"]):
            print("✗ Batch prompts are empty - encode_prompt may return unexpected results!")
        else:
            print(f"✓ Batch prompts valid: {batch['prompt']}")


class TestDebugTrainingLoop:
    """Debug version of training loop with detailed output"""
    
    @staticmethod
    def test_training_loop_debug():
        """Simulate training loop with detailed debugging"""
        print("\n=== Testing Training Loop (Debug Mode) ===")
        
        test_dir = TestDataGenerator.create_test_dataset(num_images=2)
        
        try:
            # Create dataset
            dataset = CharacterDataset(test_dir, resolution=256, dtype=torch.float32)
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            print("\nSimulating training loop:")
            for batch_idx, batch in enumerate(dataloader, 1):
                print(f"\n--- Batch {batch_idx} ---")
                
                # Check batch contents
                print(f"Batch keys: {batch.keys()}")
                print(f"pixel_values type: {type(batch['pixel_values'])}, shape: {batch['pixel_values'].shape}")
                print(f"prompt type: {type(batch['prompt'])}, value: {batch['prompt']}")
                
                # Validate pixel_values
                if batch['pixel_values'] is None:
                    print("✗ ERROR: pixel_values is None!")
                    continue
                
                # Validate prompts
                prompts = batch['prompt']
                if prompts is None:
                    print("✗ ERROR: prompts is None!")
                    continue
                
                batch_size = batch['pixel_values'].shape[0]
                
                # Handle prompts
                if isinstance(prompts, str):
                    prompts_list = [prompts] * batch_size
                elif isinstance(prompts, (list, tuple)):
                    prompts_list = list(prompts)
                else:
                    print(f"✗ ERROR: Unexpected prompts type: {type(prompts)}")
                    continue
                
                print(f"✓ Prompts converted to list: {prompts_list}")
                
                if any(p is None for p in prompts_list):
                    print("✗ ERROR: Some prompts in list are None!")
                    continue
                
                print("✓ Batch processing successful")
                
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("SDXL LoRA Training - Debug Test Suite")
    print("=" * 70)
    
    TestCharacterDataset.test_dataset_loading()
    TestPromptEncoding.test_encode_prompt_mock()
    TestVAEEncoding.test_vae_encode_mock()
    TestUNetForward.test_unet_forward_mock()
    TestBatchProcessing.test_batch_with_none_prompts()
    TestDebugTrainingLoop.test_training_loop_debug()
    
    print("\n" + "=" * 70)
    print("Test suite completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

