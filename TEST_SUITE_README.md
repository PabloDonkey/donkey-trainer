# Test Suite Documentation

## Overview
This project includes comprehensive tests to replicate and debug the `'NoneType' object has no attribute 'shape'` error encountered during LoRA training.

## Test Files

### 1. `test_training.py` - Unit Tests
Comprehensive unit tests for individual components of the training pipeline.

**Tests Included:**

#### `TestCharacterDataset.test_dataset_loading()`
- Tests image loading from disk
- Validates dataset returns proper tensors without None values
- Checks prompt/caption loading

**Run:**
```bash
python test_training.py
```

**What it tests:**
- ✓ Images load correctly
- ✓ Images resize to proper dimensions
- ✓ Tensor conversion works
- ✓ Prompts load from .txt files
- ✗ Identifies if dataset returns None values

#### `TestPromptEncoding.test_encode_prompt_mock()`
- Simulates `encode_prompt()` with mock pipeline
- Tests normal case vs. broken cases
- Shows what happens when encoder returns None

**Scenario 1 - Normal:** `encode_prompt()` returns tuple of tensors
```
✓ encode_prompt returned tuple with shapes: torch.Size([1, 77, 768]), torch.Size([1, 1280])
```

**Scenario 2 - Broken:** `encode_prompt()` returns None
```
✗ encode_prompt returned None - THIS WOULD CAUSE THE ERROR!
  Error would occur when trying: encoder_hidden_states = result[0]
```

#### `TestVAEEncoding.test_vae_encode_mock()`
- Tests VAE latent encoding
- Simulates VAE failure scenarios
- Shows how None VAE output causes AttributeError

**Scenario 1 - Normal:** VAE returns latent distribution
```
✓ VAE encoding returned tensor with shape: torch.Size([1, 4, 128, 128])
```

**Scenario 2 - Broken:** VAE returns None
```
✗ AttributeError caught: 'NoneType' object has no attribute 'latent_dist'
```

#### `TestUNetForward.test_unet_forward_mock()`
- Tests UNet forward pass
- Simulates prediction generation
- Shows impact of None UNet output

**Scenario 1 - Normal:** UNet returns prediction
```
✓ UNet forward returned tensor with shape: torch.Size([1, 4, 128, 128])
```

**Scenario 2 - Broken:** UNet returns None
```
✗ AttributeError caught: 'NoneType' object has no attribute 'sample'
```

#### `TestBatchProcessing.test_batch_with_none_prompts()`
- Tests batch data integrity
- Checks for None prompts
- Validates empty string handling

**Scenario 1 - Valid:** Normal batch with proper prompts
```
✓ Batch prompts valid: ['prompt 1', 'prompt 2']
```

**Scenario 2 - Broken:** None prompts in batch
```
✗ Batch prompt is None - THIS WOULD CAUSE ENCODE_PROMPT TO FAIL!
```

**Scenario 3 - Warning:** Empty string prompts
```
✗ Batch prompts are empty - encode_prompt may return unexpected results!
```

#### `TestDebugTrainingLoop.test_training_loop_debug()`
- Full training loop simulation
- Creates test dataset on disk
- Processes batches with detailed output
- Validates each step

**Output:**
```
--- Batch 1 ---
Batch keys: dict_keys(['pixel_values', 'prompt'])
pixel_values type: <class 'torch.Tensor'>, shape: torch.Size([1, 3, 256, 256])
prompt type: <class 'list'>, value: ['A test character number 1']
✓ Prompts converted to list: ['A test character number 1']
✓ Batch processing successful
```

---

### 2. `test_integration.py` - Integration Tests
Shows the broken training loop vs. the fixed version side-by-side.

**Tests Included:**

#### `FixedLoRATrainerTest.simulate_broken_training_loop()`
Demonstrates **3 ways** the error occurs:

**Broken Case 1: encode_prompt returns None**
```
[2] Encoding prompts (BROKEN: returns None)...
prompt_embeds = None
✗ ERROR (TypeError): 'NoneType' object is not subscriptable
  This happens because prompt_embeds is None, and None[0] throws TypeError
```

**Broken Case 2: VAE encode returns None**
```
[3] VAE Encoding (BROKEN: returns None)...
✗ ERROR (AttributeError): 'NoneType' object has no attribute 'latent_dist'
  This happens because VAE.encode() returned None
```

**Broken Case 3: UNet returns None**
```
[4] UNet Forward (BROKEN: returns None)...
✗ ERROR (AttributeError): 'NoneType' object has no attribute 'sample'
  This happens because UNet forward returned None
```

#### `FixedLoRATrainerTest.simulate_fixed_training_loop()`
Shows the **fixed version** with all validation checks:

```
[1] Processing batch with validation...
✓ Batch validated: size=1, prompts=valid

[2] Encoding prompts with validation...
✓ Prompt embeddings valid: encoder_hidden_states shape=torch.Size([1, 77, 768])

[3] VAE Encoding with validation...
✓ Latents valid: shape=torch.Size([1, 4, 128, 128])

[4] UNet Forward with validation...
✓ UNet output valid: noise_pred shape=torch.Size([1, 4, 128, 128])

[5] Loss computation...
✓ Loss valid: 0.0019

✓ FIXED TRAINING LOOP COMPLETED SUCCESSFULLY
```

#### `ProposedFix.print_fix()`
Prints the complete proposed fix code showing all 5 fixes needed in `LoRATrainer.py`:

- **FIX 1:** Validate batch contents
- **FIX 2:** Validate prompts
- **FIX 3:** Validate prompt embeddings
- **FIX 4:** Validate VAE output
- **FIX 5:** Validate UNet output

**Run:**
```bash
python test_integration.py
```

---

## Running the Tests

### Run all unit tests:
```bash
python test_training.py
```

### Run integration tests:
```bash
python test_integration.py
```

### View test logs:
```bash
cat logs/test.log
cat logs/integration_test.log
```

---

## Expected Output

### From `test_training.py`:
```
======================================================================
SDXL LoRA Training - Debug Test Suite
======================================================================

=== Testing CharacterDataset ===
✓ Dataset created with 2 images
✓ Item 0: pixel_values shape=torch.Size([3, 256, 256]), prompt='A test character number 1'
✓ Item 1: pixel_values shape=torch.Size([3, 256, 256]), prompt='A test character number 2'

=== Testing Prompt Encoding ===
Test 1: Normal encode_prompt
✓ encode_prompt returned tuple with shapes: torch.Size([1, 77, 768]), torch.Size([1, 1280])

Test 2: encode_prompt returns None
✗ encode_prompt returned None - THIS WOULD CAUSE THE ERROR!

... (more tests)

======================================================================
Test suite completed!
======================================================================
```

### From `test_integration.py`:
```
======================================================================
INTEGRATION TEST: NoneType Shape Error Replication & Fix
======================================================================

======================================================================
SIMULATING BROKEN TRAINING LOOP
======================================================================

[1] Processing batch...
✓ Batch size: 1

[2] Encoding prompts (BROKEN: returns None)...
✗ ERROR (TypeError): 'NoneType' object is not subscriptable

... (shows all 3 broken scenarios)

======================================================================
SIMULATING FIXED TRAINING LOOP WITH VALIDATION
======================================================================

[1] Processing batch with validation...
✓ Batch validated: size=1, prompts=valid

[2] Encoding prompts with validation...
✓ Prompt embeddings valid: encoder_hidden_states shape=torch.Size([1, 77, 768])

... (shows fixed version working)

✓ FIXED TRAINING LOOP COMPLETED SUCCESSFULLY

======================================================================
PROPOSED FIX FOR LoRATrainer.py
======================================================================

def train(self, epochs: int = 10, log_interval: int = 1):
    """Fixed version with proper validation and error handling"""
    # (Full fixed code shown here)
```

---

## Key Findings

### What Causes the Error:

1. **NoneType objects returned from:**
   - `pipe.encode_prompt()` returns None instead of embeddings
   - `pipe.vae.encode()` returns None
   - `unet()` forward pass returns None

2. **Accessing attributes on None:**
   - `None[0]` → TypeError: 'NoneType' object is not subscriptable
   - `None.latent_dist` → AttributeError: 'NoneType' object has no attribute 'latent_dist'
   - `None.sample` → AttributeError: 'NoneType' object has no attribute 'sample'
   - `None.shape` → AttributeError: 'NoneType' object has no attribute 'shape'

3. **Data integrity issues:**
   - Batch prompts are None
   - Batch pixel_values are None
   - Empty prompts causing encoder failures

### The Fix Strategy:

Instead of crashing when None is encountered, add validation checks and **skip problematic batches**:

```python
if variable is None:
    logger.error("Variable is None at epoch %d batch %d", epoch + 1, i)
    continue  # Skip to next batch instead of crashing
```

This allows training to continue while logging which batches had issues.

---

## Integration with LoRATrainer

To apply the fixes, the `train()` method in `LoRATrainer.py` needs these 5 validation checks inserted in the training loop.

See `test_integration.py` for the complete proposed fix code.

---

## Requirements

- PyTorch
- diffusers
- peft
- PIL
- numpy
- torch.utils.data.DataLoader

All should be available in your existing environment.

