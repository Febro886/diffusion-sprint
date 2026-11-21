# -*- coding: utf-8 -*-
r"""从训好的蝴蝶 DDPM 当场生成一张 4x4 拼图，存成 PNG 文件。
用法（新开一个 Anaconda Prompt，不要动正在训练的窗口）：
  1. conda activate diffusion
  2. cd C:\Users\lenovo\diffusion-sprint\diffusion-train
  3. 把这个文件存进这个文件夹，名字就叫 generate_butterfly_samples.py
  4. python generate_butterfly_samples.py
出图：当前文件夹里的 butterfly_samples.png（16 只新蝴蝶，4x4 拼图）。
原理：训练结束时脚本已把整条 pipeline 存进 ddpm-butterflies-64，
这里直接加载它，从纯噪声采样 50 步生成新图。种子固定 42，
每次跑出来的 16 只都一样，方便对照。
"""
import torch
from diffusers import DDPMPipeline
from PIL import Image

pipe = DDPMPipeline.from_pretrained("ddpm-butterflies-64").to("cuda")
pipe.set_progress_bar_config(disable=True)

images = pipe(
    batch_size=16,
    num_inference_steps=50,
    generator=torch.Generator("cuda").manual_seed(42),
).images

w, h = images[0].size
grid = Image.new("RGB", (w * 4, h * 4), "white")
for i, im in enumerate(images):
    grid.paste(im, ((i % 4) * w, (i // 4) * h))
# 原图只有 64x64，放大 4 倍再存，看得清楚
grid = grid.resize((w * 16, h * 16), Image.NEAREST)
grid.save("butterfly_samples.png")
print("saved butterfly_samples.png, 16 images, seed 42")
