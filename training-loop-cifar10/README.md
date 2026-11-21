# 手写训练循环（CIFAR-10）+ 手写 Attention

- `train.py`：从零手写的完整训练循环（数据加载、模型、损失、优化器、训练/评估、保存），CIFAR-10 测试准确率 78.8%，曲线见 `acc_curve.png`。
- `attention.py`：手写 scaled dot-product self-attention，并与 PyTorch 官方 `nn.MultiheadAttention` 对照验证，输出一致。
- `checkpoint.pth`：训练得到的模型权重。
