[第一版8.7.md](https://github.com/user-attachments/files/30819948/8.7.md)
### 1 引言

1. AIGC视频生成技术发展；（介绍ChatGPT、Sora等推动AIGC快速发展）
2. 应用价值与安全风险；（说明伪造内容对个人、企业和国家造成的风险）
3. 从局部Deepfake向整体视频生成的变化；
4. 现有检测方法面临的问题；
5. 已有综述的不足；
6. 本文分类体系和贡献。

​	生成式人工智能正在改变视频的生产方式。早期合成视频主要围绕换脸、表情迁移和口型驱动展开，通常以真实视频为载体并修改局部区域；视频扩散模型、DiT和大规模视频基础模型则能够从文本、图像或多模态条件直接合成完整片段。随着生成质量提高，视频证据所承载的“谁、何时、何地、做了什么”不再天然可信。由此产生的风险既包括个人层面的冒名、诈骗和隐私侵害，也包括企业声誉攻击、市场操纵以及公共安全与舆论治理风险[1-3]。

​	AIGC生成视频检测与传统Deepfake检测既连续又不同。二者都研究机器合成或操纵痕迹，但传统任务多围绕人脸局部篡改，依赖融合边界、几何异常、生理信号或相机噪声；端到端生成视频可能没有真实载体，也未必留下固定的局部边界，决定性证据更可能来自跨帧运动、物体恒常性、物理规律、音画关系乃至外部事实[1,4-7]。Vahdati等发现，将合成图像检测器直接迁移到生成视频时效果有限，说明视频生成器留下的取证痕迹具有独特性[4]。DeCoF和DeMamba则分别从帧一致性和时空细节建模入手，推动检测从逐帧判别转向视频级证据聚合[5-6]。

​	已有Deepfake综述系统总结了人脸操纵、媒体取证与可靠性问题[2-3,8-9]，但对端到端生成视频、跨模态语义核验、MLLM解释和智能体工具调用覆盖不足。Hou等提出视觉—语言双视角和四层分类，将AIGC视频检测重述为“事实保真验证”[1]。本文沿用并扩展这一思路，以用户提供的第一版提纲为结构基础，强调“证据层级”而非“Non-MLLM/MLLM”作为一级分类。本文贡献包括：一是统一界定生成范式和检测任务；二是按由低层感知到高层认知的证据链综述方法；三是将数据、指标与协议对应到真实部署风险；四是讨论被动检测与主动认证的协同边界。

### 2 AIGC生成视频的范围与生成范式

- 局部操纵视频；
- 音视频编辑；
- 端到端生成视频；
- 明确区分Deepfake Video与AIGC Generated Video。

### 3 检测任务定义与统一框架

- 真实性分类；
- 时空定位；
- 伪造来源归因；
- 检测解释；
- 事实保真度验证。

### 4 AIGC视频检测

### 4.1 基于视觉内在痕迹的检测

- 空域和几何伪影；
- 频域和噪声特征；
- 生理信号；
- 生成模型指纹；
- 表征分布差异。

### 4.2 基于时空一致性的检测

- 帧间不一致；
- 光流与运动轨迹；
- 长时依赖；
- 物理规律；
- 人体行为和交互动态。

### 4.3 基于跨模态一致性的检测

- 音频—视觉同步；
- 身份—声纹一致性；
- 文本—视频语义一致性；
- 多模态时间定位。

### 4.4 基于VLM、MLLM和智能体的检测

- 提示工程与零样本检测；
- 参数高效微调；
- 外接专业检测器；
- 检索增强与工具调用；
- 世界知识和事实验证；
- 证据生成与自然语言解释。

### （5 主动防御技术（在前言中简要介绍））

### 5 数据集、评测指标与实验协议

- 按三种生成范式整理数据集；（局部操纵视频；音视频编辑；端到端生成视频；）
- 帧级、视频级和片段级指标；（检测指标；干扰指标；认证指标。）（通用二分类指标；视觉视角指标；语言视角指标；定位、解释和事实验证指标。）
- 跨数据集测试；
- 跨生成器测试；
- 压缩和社交媒体退化测试；
- 对抗鲁棒性；
- 定位和解释质量评测。

### 6 可信性、局限与未来方向

- 未知生成器泛化；
- 开放集检测；
- 后处理和对抗攻击；
- 模型幻觉与错误解释；（统一的可解释检测）
- 证据校准和不确定性；（ 证据优先的可信检测）
- 计算效率；
- 内容检测与水印、来源认证协同；
- 动态更新基准。

### 7 结论

概括检测证据由低层伪影向高层事实验证演进的趋势。

最关键的结构选择是：**不要把Non-MLLM与MLLM作为全文的一级并列分类，而应以检测证据层级作为一级分类，再在每一类方法中标注其属于传统模型、视觉基础模型还是MLLM。**这样能减少重复，也更能体现AIGC视频检测从感知取证向语义推理发展的技术脉络。

### 8 参考文献

[1] Hou D X, Zhang J, Gu X, et al. Detecting AI-Generated Video: A Vision-Language Dual-View Survey[C]//Findings of the Association for Computational Linguistics: ACL 2026. 2026: 32221-32255. DOI: 10.18653/v1/2026.findings-acl.1613.

[2] Mirsky Y, Lee W. The creation and detection of deepfakes: A survey[J]. ACM Computing Surveys, 2021, 54(1): 1-41. DOI: 10.1145/3425780.

[3] Verdoliva L. Media forensics and deepfakes: An overview[J]. IEEE Journal of Selected Topics in Signal Processing, 2020, 14(5): 910-932. DOI: 10.1109/JSTSP.2020.3002101.

[4] Vahdati D S, Nguyen T D, Azizpour A, et al. Beyond Deepfake Images: Detecting AI-Generated Videos[C]//CVPR Workshops. 2024: 4397-4408.

[5] Ma L, Zhang J, Deng H, et al. DeCoF: Generated Video Detection via Frame Consistency: The First Benchmark Dataset[EB/OL]. arXiv:2402.02085, 2024.

[6] Chen H, Hong Y, Huang Z, et al. DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark[EB/OL]. arXiv:2405.19707, 2024.

[7] Ni Z, Zhai Y, Ma J, et al. A Challenging Benchmark for Detecting AI-Generated Video[EB/OL]. arXiv:2501.11340, 2025.

[8] Tolosana R, Vera-Rodriguez R, Fierrez J, et al. Deepfakes and beyond: A survey of face manipulation and fake detection[J]. Information Fusion, 2020, 64: 131-148. DOI: 10.1016/j.inffus.2020.06.014.

[9] Wang T, Chow K P. Deepfake detection: A comprehensive survey from the reliability perspective[J]. ACM Computing Surveys, 2024.

[10] Rossler A, Cozzolino D, Verdoliva L, et al. FaceForensics++: Learning to Detect Manipulated Facial Images[C]//ICCV. 2019: 1-11. DOI: 10.1109/ICCV.2019.00009.

[11] Li Y, Yang X, Sun P, et al. Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics[C]//CVPR. 2020: 3207-3216. DOI: 10.1109/CVPR42600.2020.00327.

[12] Dolhansky B, Bitton J, Pflaum B, et al. The DeepFake Detection Challenge (DFDC) Dataset[EB/OL]. arXiv:2006.07397, 2020.

[13] He Y, Gan B, Chen S, et al. ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis[C]//CVPR. 2021: 4360-4369. DOI: 10.1109/CVPR46437.2021.00434.

[14] Khalid H, Tariq S, Kim M, et al. FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset[EB/OL]. arXiv:2108.05080, 2021.

[15] Cai Z, Ghosh S, Stefanov K, et al. Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization[C]//DICTA. 2022. DOI: 10.1109/DICTA56598.2022.10034605.

[16] Cai Z, Stefanov K, Dhall A, et al. AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset[C]//ACM Multimedia. 2024. DOI: 10.1145/3664647.3680795.

[17] Ho J, Chan W, Saharia C, et al. Video Diffusion Models[EB/OL]. arXiv:2204.03458, 2022.

[18] Ho J, Salimans T, Gritsenko A, et al. Imagen Video: High Definition Video Generation with Diffusion Models[EB/OL]. arXiv:2210.02303, 2022.

[19] Singer U, Polyak A, Hayes T, et al. Make-A-Video: Text-to-Video Generation without Text-Video Data[EB/OL]. arXiv:2209.14792, 2022.

[20] Li H, Wang B, Hu Z, et al. FreqBlender: Enhancing DeepFake Detection by Blending Frequency Knowledge[C]//Advances in Neural Information Processing Systems. 2024, 37.

[21] Wang T, Chow K P. Noise Based Deepfake Detection via Multi-Head Relative-Interaction[C]//AAAI. 2023, 37(12): 14548-14556. DOI: 10.1609/aaai.v37i12.26701.

[22] Qi H, Guo Q, Juefei-Xu F, et al. DeepRhythm: Exposing DeepFakes with Attentional Visual Heartbeat Rhythms[C]//ACM Multimedia. 2020: 4318-4327. DOI: 10.1145/3394171.3413707.

[23] Song X, Guo X, Zhang J, et al. On Learning Multi-Modal Forgery Representation for Diffusion Generated Video Detection[C]//Advances in Neural Information Processing Systems. 2024, 37: 122054-122077.

[24] Park K, Yang Y, Yi J, et al. VidGuard-R1: AI-Generated Video Detection and Explanation via Reasoning MLLMs and RL[EB/OL]. arXiv:2510.02282, 2025.

[25] Chen W, Zheng W, Zheng Y, et al. GenWorld: Towards Detecting AI-generated Real-world Simulation Videos[EB/OL]. arXiv:2506.10975, 2025.

[26] Coalition for Content Provenance and Authenticity. C2PA Technical Specification, Version 2.4[EB/OL]. 2026. https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html.

[27] Yan Z, Zhang Y, Yuan X, et al. DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection[C]//Advances in Neural Information Processing Systems, Datasets and Benchmarks Track. 2023, 36.

[28] Ma L, Yan Z, Guo Q, et al. Your One-Stop Solution for AI-Generated Video Detection[C]//CVPR. 2026. arXiv:2601.11035.

[29] Yan Z, Zhao Y, Chen S, et al. Generalizing Deepfake Video Detection with Plug-and-Play: Video-Level Blending and Spatiotemporal Adapter Tuning[C]//CVPR. 2025: 12615-12625.

[30] Sun H, Cai C, Zhuang H, et al. EDVD-LLaMA: Explainable Deepfake Video Detection via Multimodal Large Language Model Reasoning[EB/OL]. arXiv:2510.16442, 2025.
