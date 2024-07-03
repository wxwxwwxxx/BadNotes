# Continual Self-supervised Learning: Towards Universal Multi-modal Medical Data Representation Learning
## 思路
混合多个模态进行预训练。单纯的混合并训练会有significant performance drop，作者称之为modal data collision。这是由于不同模态之间的差异使得表征学习存在冲突。同时，先前的旧模态需要复习，会增加训练成本。

作者使用了Sequential SSL，这是一种基于Continual SSL的想法，模型逐个学习模态，并在学习时不断Rehearsal 5%之前的模态。此外，作者对特征进行了Feature distillation，并对Rehearsal模态进行了Intra-modal mixup的增强。

## 其他结论
- 作者在Related Works中整理了两种不同的多模态自监督学习方案
    - Joint SSL: 混合不同模态
    - Sequential SSL: 每个模态依次序逐个学习，并复习

- 几种不同的自监督范式
    - Single-modal SSL
        - 使用Masked Modeling学习
    - Paired multi-modal SSL
        - 如对比学习
    - Unpaired multi-modal SSL
        - 不使用成对的样本，有着更低的data constraints
        - 更容易scaling

注：Unpaired multi-modal SSL对于使用医院数据训练具有比较重要的意义

## Links
- [code](https://github.com/yeerwen/MedCoSS)
- [paper](https://arxiv.org/pdf/2311.17597)
- [微信介绍链接](https://mp.weixin.qq.com/s/Ul_KiOUXo9Fy6AkNb_C_NQ)