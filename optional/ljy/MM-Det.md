这篇论文发表于 NeurIPS 2024，方法名为 **MM-Det**，数据集名为 **DVF（Diffusion Video Forensics）**。它的核心思想可以压缩成一句话：

> **不要只寻找像素层面的生成痕迹，而是同时利用大模型对“不自然内容”的语义理解，以及视频在空间、时间和重建过程中的技术性异常。**

作者认为现有问题：1，人脸伪造检测的任务范围过窄；2，图像生成检测忽略时间维度；3，检测器容易记住特定生成模型。
给定一段视频，判断它是不是由扩散式生成模型完整合成的。
MM-Det由两条并行分支和一个融合模块组成：

1. **LMM分支**：利用大视觉语言模型提取多模态伪造表征 MMFR；
2. **ST分支**：利用重建差异和时空注意力提取技术性伪造痕迹；
3. **动态融合模块**：根据不同视频，自适应决定两类证据各占多大权重。
<img width="1491" height="1055" alt="image" src="https://github.com/user-attachments/assets/6339b1ad-06cb-4d6a-a6ed-98d4ec45d51d" />

<img width="1448" height="1086" alt="image" src="https://github.com/user-attachments/assets/e34e652d-787c-47a0-86bd-70296655b43b" />

[SparkleXFantasy/MM-Det: Diffusion Generated Video Detection (NeurIPS2024)](https://github.com/SparkleXFantasy/MM-Det?utm_source=chatgpt.com)

[sparklexfantasy/DVF · Datasets at Hugging Face](https://huggingface.co/datasets/sparklexfantasy/DVF?utm_source=chatgpt.com)

代码运行结果：
<img width="1265" height="1401" alt="image" src="https://github.com/user-attachments/assets/3d28a613-cd6c-4e37-bcfa-a9e0016b70f8" />

MM-Det的训练配置：
真实视频：1,000 个 YouTube 视频
伪造视频：1,800 个 Stable Video Diffusion 视频
划分方式：80% 训练，20% 验证
<img width="870" height="747" alt="image" src="https://github.com/user-attachments/assets/4495584b-5ff4-4d3a-a600-400fa9be9f05" />
