import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model

# Set environment variable to disable Flax
os.environ["DIFFUSERS_NO_FLAX"] = "1"

# Parameters
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DATA_DIR = "data/face_images"
CSV_PATH = "data/metadata.csv"
OUTPUT_DIR = "output/lora_weights"
BATCH_SIZE = 1
NUM_STEPS = 1000  # Increased from 500
LEARNING_RATE = 5e-5
LORA_RANK = 16  # Increased from 4

class FaceDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.df = pd.read_csv(csv_path)
        print(f"CSV loaded with {len(self.df)} entries:")
        print(self.df)
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.df.iloc[idx]["file_name"])
        try:
            image = Image.open(img_name).convert("RGB")
            print(f"Success: {self.df.iloc[idx]['file_name']} is readable")
        except Exception as e:
            print(f"Error: {self.df.iloc[idx]['file_name']} cannot be opened: {str(e)}")
            raise
        image = self.transform(image)
        caption = self.df.iloc[idx]["text"]
        return image, caption

def train():
    # Check device and set dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Explicitly set to float32 to avoid nan issues
    print(f"Running on device: {device} with dtype: {dtype}")

    # Load dataset
    dataset = FaceDataset(CSV_PATH, DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset size: {len(dataset)}, Batches per epoch: {len(dataloader)}")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe.to(device)

    # Extract UNet and noise scheduler
    unet = pipe.unet
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=16,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"]
    )
    unet = get_peft_model(unet, lora_config)

    # Prepare for training
    unet.train()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    accelerator = Accelerator(mixed_precision="no")  # Disable mixed precision
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    # Load text encoder and tokenizer
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    text_encoder.eval()

    # Training loop
    for step, batch in enumerate(range(NUM_STEPS)):
        images, captions = next(iter(dataloader))
        images = images.to(device)

        # Encode images to latents
        latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
        print(f"Step {step}: Images shape: {images.shape}, Captions: {captions}")

        # Tokenize captions
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(text_input_ids)[0]
        print(f"Step {step}: Text embeddings shape: {text_embeddings.shape}")

        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)
        print(f"Step {step}: Latents shape: {latents.shape}")

        # Forward pass
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
        print(f"Step {step}: Model prediction shape: {model_pred.shape}")

        # Compute loss
        target = noise
        loss = torch.nn.functional.mse_loss(model_pred, target)
        accelerator.backward(loss)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Log
        print(f"Step {step}/{NUM_STEPS}, Loss: {loss.item()}")

    # Save LoRA weights
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unet.save_pretrained(OUTPUT_DIR, safe_serialization=True)  # Save as .safetensors
    unet.save_pretrained(OUTPUT_DIR, safe_serialization=False)  # Save as .bin
    print(f"LoRA weights saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()