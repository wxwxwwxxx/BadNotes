# STU-Net: Scalable and Transferable Medical Image Segmentation Models Empowered by Large-Scale Supervised Pre-training
## 思路
修改了nnU-net的升采样层，降采样层，增加了residual block，使其结构的深度和宽度可拓展。同时由于消除了与模型权重相关的超参数，使得预训练->下游任务Fine-tune成为可能，从而使模型可以适配多种数据集、任务及模态。
在此基础上，作者提出了1.4B的医学影像大模型，并得出结论：大尺度模型在各种数据量的情况下均相对于小尺度模型有性能提升。

## 其他结论
- 预训练数据：TotalSegmentor
- 同样采用了2的batch size
- 镜像的数据增强在预训练阶段和下游任务Fine-tune阶段均有帮助
- 论文的实验指出，基于Transformer的模型在同样的参数量下，均差于基于U-net的模型
- U-net的深度（层数）和宽度（通道数）应该一同拓展（EfficientNet的结论）
- 上采样中，最近邻（NN）的表现优于三线性插值，计算消耗也低于三线性插值
- 大尺度模型在各种数据量的情况下均相对于小尺度模型有性能提升（重要），而且更多的数据量似乎能让模型的性能进一步优化
- 随着模型尺度变大，通用模型（分割所有104个标签的模型）效果会更好
- 相对于随机初始化的输出层，预训练的参数可以使用更小的学习率，论文中明文说明了这种做法会让结果更好
- 下游任务输入通道数与预训练模型不匹配的情况下，可以通过复制预训练模型的输入层来实现
- 引用[4,21]关于基础模型，Efficientnet[35]关于模型结构设计（shufflenet等也相关），[20,42]和模型的scaling有关

## Links
- [code](https://github.com/uni-medical/STU-Net)
- [paper](https://arxiv.org/abs/2304.06716)