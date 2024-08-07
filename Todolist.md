待学习列表:
- MLLM
    - [LLaVa](https://github.com/haotian-liu/LLaVA)
    - [Video-LLaVa](https://github.com/PKU-YuanGroup/Video-LLaVA)
    - [LLava-Med](https://github.com/microsoft/LLaVA-Med)
    - [NExT-GPT](https://github.com/NExT-GPT/NExT-GPT)
    - [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)
    - [InternLM-XComposer](https://github.com/InternLM/InternLM-XComposer)
    - [CogVLM](https://arxiv.org/abs/2311.03079)
    - 其他可参考[排行榜](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard)

- LLM
    - Model
        - [phi-2](https://huggingface.co/microsoft/phi-2)
        - [LLaMa](https://github.com/facebookresearch/llama)
        - [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf)
        - [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
        - 其他可参考[排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)以及[Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM)
    - Training or Inference Technique
        - 量化
            - [bitnet b1.58](https://arxiv.org/abs/2402.17764)
            - [What are Quantized LLMs?](https://www.tensorops.ai/post/what-are-quantized-llms)
        - RLHF, [项目集合](https://github.com/opendilab/awesome-RLHF), [博客](https://huggingface.co/blog/zh/rlhf)
        - RAG, [Langchain实现](https://python.langchain.com/docs/use_cases/question_answering/), [Survey](https://arxiv.org/abs/2312.10997)
        - [LoRA](https://arxiv.org/abs/2106.09685)
        - [Awesome-LLM-Inference](https://github.com/DefTruth/Awesome-LLM-Inference)
        - [Continuous Batching](https://zhuanlan.zhihu.com/p/676109470)
        - KV Cache

- Deployment
    - [TensorRT](https://github.com/NVIDIA/TensorRT)
    - [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

- Vector Database
    - [Faiss](https://github.com/facebookresearch/faiss)

- Diffuser
    - [DDPM](https://arxiv.org/abs/2006.11239)
    - [DDIM](https://arxiv.org/abs/2010.02502)
    - [Diffusers](https://github.com/huggingface/diffusers)
    - [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
    - [TextDiffuser](https://arxiv.org/abs/2305.10855), [TextDiffuser2](https://arxiv.org/abs/2311.16465)

- Pretrain Related
    - VQ-VAE
        - [VQ-VAE](https://arxiv.org/pdf/1711.00937.pdf)
        - [VQ-VAE2](https://arxiv.org/pdf/1906.00446.pdf)
        - [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505)

    - Medical Pretrain Related
        - [STU-Net](https://github.com/uni-medical/STU-Net)

- Training Technique, Tricks
    - 混合精度, torch.amp[教程](https://pytorch.org/docs/stable/notes/amp_examples.html)
    - torch.complie
    - torchsript/JIT, [Demo](https://github.com/louis-she/torchscript-demos)
    - [DeepSpeed](https://github.com/microsoft/DeepSpeed)

- Model Improvement, Efficient
    - [Transformer-VQ](https://spaces.ac.cn/archives/9844)

- Model Improvement, Performance
    - DropPath
    - RandAugment
    - LayerScale
    - [RoPE](https://www.zhihu.com/tardis/zm/art/647109286?source_id=1003)
    - [BEVFormer](https://arxiv.org/abs/2203.17270)
    - [Mamba](https://arxiv.org/abs/2312.00752), [OpenReview](https://openreview.net/forum?id=AL1fq05o7H)
    - [TTT](https://arxiv.org/pdf/2407.04620)

- Theory
    - [浅谈Transformer的初始化、参数化与标准化](https://spaces.ac.cn/archives/8620)
    - 关于[互信息](https://zh.wikipedia.org/wiki/%E4%BA%92%E4%BF%A1%E6%81%AF)
    - [卷积](https://zh.wikipedia.org/wiki/%E5%8D%B7%E7%A7%AF)

- Long-Tail Learning
    - [Generalized Logit Adjustment](https://zhuanlan.zhihu.com/p/548735583)
    - [Logit Adjustment](https://arxiv.org/pdf/2007.07314.pdf)

- Classical ML
    - [博客](https://www.zhihu.com/people/xu-tao-83-44-34/posts)
    - 概率图模型, [博客](https://longaspire.github.io/blog/%E6%A6%82%E7%8E%87%E5%9B%BE%E6%A8%A1%E5%9E%8B%E6%80%BB%E8%A7%88/), 以及周志华《机器学习》

- 源码阅读
    - [Pytorch](https://github.com/pytorch/pytorch?tab=readme-ov-file)
        - [Contribution Page](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md)
        - [Contribution Guide](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions)
    - [timm](https://github.com/huggingface/pytorch-image-models)

- Engineering
    - Software Related
    - [Clang](https://clang.llvm.org/docs/UsersManual.html)
    - [llvm](https://llvm.org/docs/)
    - [计算机体系结构基础](https://foxsen.github.io/archbase/)
    - Low-Level and Hardware Related
        - [GPU卡的底层通信原理](https://www.jianshu.com/p/e40059d5c832)

- HPC/Computer Architecture/CUDA
    - [CMU15-418](https://www.bilibili.com/video/BV1qT411g7n9/?spm_id_from=333.337.search-card.all.click&vd_source=48595d95206943f17002312e5fdd961f)
    - [计算机体系结构基础](https://foxsen.github.io/archbase/index.html)
    - [CMU18-447](https://www.bilibili.com/video/BV1PT4y1M7gM/?spm_id_from=333.999.0.0&vd_source=48595d95206943f17002312e5fdd961f)
    - [Computer Architecture ETH Zurich](https://www.bilibili.com/video/BV1Vf4y1i7YG/?spm_id_from=333.999.0.0&vd_source=48595d95206943f17002312e5fdd961f)
    - CUDA并行程序设计 GPU编程指南
    - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)

- Writing
    - [Paper Writing Tips](https://github.com/MLNLP-World/Paper-Writing-Tips)
    - [如何端到端地写科研论文？](http://www.cips-cl.org/static/CCL2018/downloads/stuPPT/qiuxp.pdf)
        - Abstract
            - 问题是什么？
            - 我们要做什么？
            - 大概怎么做的？
            - 做得还不错!
        - Introduction
            - 起：问题是什么？
            - 承：相关工作
            - 转：相关工作的不足与转机
            - 合：本文工作、本文贡献（通常分点）
        - Method
            - 简要的重复问题
            - 解决思路
            - 形式化定义
            - 具体模型

- Miscellaneous
    - [复杂性的那些事](https://www.zhihu.com/column/c_1389404662173315072)
    - [Antkillerfarm Hacking V6.5](https://antkillerfarm.github.io/)
    - [科学空间](https://spaces.ac.cn/)
    - [Raphaël Millière](https://raphaelmilliere.com/)
    - 一些知乎用户
        - [链接](https://www.zhihu.com/people/xia-jing-jing-57/)
        - [链接](https://www.zhihu.com/people/liu-dong-13)
    - [OpenGVLab](https://github.com/OpenGVLab)
    - OpenMMLab, [Github](https://github.com/open-mmlab), [HomePage](https://openmmlab.com/)
    - [adam在大模型预训练中的不稳定性分析及解决办法](https://zhuanlan.zhihu.com/p/675421518)
- Books
    - Computer Organization and Architecture: Themes and Variations
    - 现代操作系统
    - 概率论 茆诗松
    - 数理统计与数据分析 John A. Rice
        - 统计学书籍比较多，下列书籍可以作为参考：
        - 统计推断 George Casella
        - 统计学习基础 -数据挖掘，推理与预测 Trevor Hastie
    - 机器学习 周志华
    - 统计学习方法 李航

- Interview Experience
    - [1](https://www.zhihu.com/question/426238388/answer/2937544836)
    - [2](https://zhuanlan.zhihu.com/p/690474151)
    - [如何判断候选人有没有千卡GPU集群的训练经验？](https://www.zhihu.com/question/650979052)