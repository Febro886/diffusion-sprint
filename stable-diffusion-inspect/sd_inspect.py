from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.enable_attention_slicing()
print("VAE:", pipe.vae.config)
print("UNet:", pipe.unet.config)
print("TextEncoder:", pipe.text_encoder.config)
print("Scheduler:", pipe.scheduler.config)