# SD 组件笔记（2026-11-19，SD v1.5 实测配置）

## VAE（AutoencoderKL）
- 吃什么：512×512×3 的 RGB 图（编码时）/ 64×64×4 潜变量（解码时）
- 吐什么：64×64×4 潜变量 / 还原出的图；关键数值：latent_channels=4，scaling_factor=0.18215
- 坏了会怎样：图糊、颜色发灰发怪

## UNet（UNet2DConditionModel）
- 吃什么：4 通道潜变量 + 时间步 + CLIP 给的文本向量（cross_attention_dim=768）
- 吐什么：4 通道的预测噪声（epsilon）
- 坏了会怎样：生成不出有意义，有内容的图

## CLIP 文本编码器（CLIPTextModel）
- 吃什么：一句提示词，最多切成 77 个 token
- 吐什么：77×768 的文本向量；关键数值：hidden_size=768，12 层、12 头
- 坏了会怎样：图文对不上

## 调度器（PNDMScheduler）
- 吃什么：UNet 每步预测的噪声 + 噪声日程（训练 1000 步，beta 从 0.00085 到 0.012，scaled_linear）
- 吐什么：一步步去噪后的潜变量；推理时把 1000 步压缩到约 50 步
- 坏了会怎样：图洗不干净

## 一句话总结
四个组件的接口是咬合的：VAE 的 latent_channels 4 = UNet 的 in_channels 4；CLIP 的 hidden_size 768 = UNet 的 cross_attention_dim 768。扩散发生在 VAE 压出的潜空间里，UNet 每步预测噪声，调度器排班，CLIP 负责让图听懂人话。
