# LoRA 微调 SD 1.5（宝可梦）

- 数据集：reach-vb/pokemon-blip-captions（公开镜像；原 lambdalabs 版已改为需申请）
- 参数：rank 4、batch 1 × 梯度累积 4、500 步、lr 1e-4、bf16、梯度检查点、单卡 RTX 4060 8GB
- 结果：loss 约 0.13 → 0.002，耗时 18 分 26 秒；前后对比见 lora_before.png / lora_after.png（同提示词同种子，after 风格迁移明显）
- 踩坑：fp16 会报梯度缩放错误，4060 改用 bf16
