import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
from peft import PeftModel

# Set environment variable to disable Flax (consistent with training)
os.environ["DIFFUSERS_NO_FLAX"] = "1"

# Parameters
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LORA_WEIGHTS = os.path.join(BASE_DIR, "output", "lora_weights")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "generated_images")
PERSON_NAME = "sks"  # Matches identifier in metadata.csv

# Prompts for watercolor-style scenes with added clarity keywords
PROMPTS = [
    f"A watercolor painting of {PERSON_NAME} in a futuristic spacesuit, standing on a moon landscape, vibrant colors, detailed textures, high resolution, sharp details",
    f"A watercolor painting of {PERSON_NAME} riding a horse in a lush meadow, dynamic pose, soft pastel tones, high resolution, sharp details",
    f"A watercolor painting of {PERSON_NAME} playing cricket on a green field, action pose, bright and vivid colors, high resolution, sharp details",
]
NEGATIVE_PROMPT = "blurry, low quality, distorted, extra limbs, unnatural colors"

def generate_images():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check device and set dtype (consistent with training)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # Use the same precision as training
    print(f"Running on device: {device} with dtype: {dtype}")

    # Load Stable Diffusion pipeline
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            safety_checker=None  # Disable safety checker to avoid potential issues
        )
    except Exception as e:
        print(f"Failed to load pipeline: {str(e)}")
        raise

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Wrap the UNet with PeftModel to load LoRA weights
    print(f"Loading LoRA weights from: {LORA_WEIGHTS}")
    try:
        # Replace the pipeline's UNet with a PeftModel-wrapped UNet
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_NAME,
            subfolder="unet",
            torch_dtype=dtype
        )
        unet = PeftModel.from_pretrained(unet, LORA_WEIGHTS)
        pipe.unet = unet
        print("LoRA weights loaded successfully")
    except Exception as e:
        print(f"Failed to load LoRA weights: {str(e)}")
        raise

    pipe.to(device)

    # Generate images with increased inference steps and guidance scale
    for i, prompt in enumerate(PROMPTS):
        try:
            image = pipe(
                prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=75,  # Increased from 50
                guidance_scale=10.0,  # Increased from 7.5
            ).images[0]
            output_path = os.path.join(OUTPUT_DIR, f"scene_{i+1}.png")
            image.save(output_path)
            print(f"Generated image saved to {output_path}")
        except Exception as e:
            print(f"Failed to generate image {i+1}: {str(e)}")

if __name__ == "__main__":
    generate_images()