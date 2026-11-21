import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

prompt = "A pokemon with blue eyes."

image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
generator=torch.Generator("cuda").manual_seed(42)).images[0]
image.save("lora_before.png")

pipe.load_lora_weights("pokemon-lora")

image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
generator=torch.Generator("cuda").manual_seed(42)).images[0]
image.save("lora_after.png")