# Diffusion Sprint — 扩散模型冲刺实录

在单卡 RTX 4060（8GB）上，从零跑通扩散模型的学习与训练全流程：手写训练循环、拆解 Stable Diffusion、真实训练一个 DDPM、再用 LoRA 微调 SD 1.5。全部代码、实验图与记录都在本仓库。

## 目录

| 文件夹 | 内容 | 关键结果 |
| --- | --- | --- |
| `env-quickstart` | PyTorch 官方 quickstart 跑通，验证 CUDA 环境 | 环境就绪（PyTorch 2.14.0+cu126） |
| `training-loop-cifar10` | 手写完整训练循环（CIFAR-10）+ 手写 self-attention 并与官方实现对照 | 测试准确率 78.8%；attention 输出与 `nn.MultiheadAttention` 一致 |
| `stable-diffusion-inspect` | SD 1.5 四大组件拆解（CLIP / VAE / UNet / Scheduler）+ 步数与引导强度对比实验 | 25 步后收益递减；guidance 过低会主体偏移（见文件夹内 notes.md 与对比图） |
| `diffusion-train` | DDPM 蝴蝶生成：第一次真实扩散训练 | loss 0.173 → 0.005（20 epoch / 2500 步，约 13 分钟），样品见 `butterfly_samples.png` |
| `lora-finetune` | LoRA 微调 SD 1.5（宝可梦风格） | loss 约 0.13 → 0.002（500 步，18 分 26 秒），前后对比图在文件夹内 |

## 环境

- GPU：NVIDIA GeForce RTX 4060 Laptop（8GB），CUDA 12.7
- PyTorch 2.14.0+cu126、diffusers 0.43.0、transformers、accelerate、peft
