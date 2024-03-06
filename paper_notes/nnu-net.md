# nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
## 思路
通过预定义规则，自动实现U-net对任意医学分割数据集的适配。

规则分为**fixed parameters**, **rule-based parameters**和**empirical parameters**三种，**fixed parameters**多指优化器、学习率等超参数，在nnU-net中均为固定参数。**Rule-based**为重采样细节、patch size等与数据集和GPU显存大小有关的细节。文中有详细的设计思路，非常有参考价值。**empirical parameters**为后处理中模型集成相关的处理，通常是按照实验效果确定的。

## 其他结论
- 模型结构的影响小于模型配置（特别是rule-based parameters）的影响
- cascade U-net可用于解决patch-size带来的上下文-清晰度之间的矛盾
- 归一化方式：其他模态依照图片进行z-score。CT在数据集层面进行**全局z-score**，这是由于CT值对于分辨器官有意义
- 后处理可以去除最大区域外的连通区域，可参考文章的18和25引用，以及method的empirical parameters中的后处理部分
- 模型集成方案：对多个模型的softmax取均值
- 本文方案的搜索空间小于AutoML，同时规则引入的归纳偏置可以帮助医学影像分割问题
- 本文的Batch size最小可以为2。此时归一化层采用instance normalization，同时需要较大的优化器动量