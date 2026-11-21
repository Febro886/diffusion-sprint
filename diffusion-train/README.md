# DDPM 蝴蝶生成（第一次真实扩散训练）

- 数据集：huggan/smithsonian_butterflies_subset（约 1000 张，64×64）
- 参数：batch 8、20 epoch、lr 1e-4、fp16、单卡 RTX 4060 8GB
- 结果：loss 0.173 → 0.005，step 2500/2500，耗时约 13 分钟；样品见 butterfly_samples.png（训后模型生成，种子 42）
- 踩坑：官方脚本须用与 diffusers 0.43.0 对版的版本；训练样品图只进 TensorBoard 不存文件，故另写 generate_butterfly_samples.py 出图
