import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
prompt = "a photo of a butterfly on a flower"
for steps in [10, 25, 50]:
    image = pipe(prompt, num_inference_steps=steps, guidance_scale=7.5,
                 generator=torch.Generator("cuda").manual_seed(42)).images[0]
    image.save(f"sd_steps_{steps}.png")
image = pipe(prompt, num_inference_steps=25, guidance_scale=2.0,
             generator=torch.Generator("cuda").manual_seed(42)).images[0]
image.save("sd_guidance_2.png")