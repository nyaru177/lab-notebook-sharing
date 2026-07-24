

### 1. VFM：视觉基础模型迁移路线

#### 1.1 视觉基础表征迁移方法（Visual Foundation Representation Transfer）

**核心定义**：利用大规模视觉或视频预训练模型作为特征提取底座，提升 VAD 在复杂场景、少样本和跨域条件下的视觉表征泛化能力。

#### 1.2 时序定位增强方法（Temporal Localization Enhanced VFM）

**核心定义**：在视觉基础模型之上引入时序 Transformer、时空注意力、长期上下文聚合或局部-全局动态建模，以增强异常片段的时间定位和动态变化感知能力。

#### 1.3 生成式自监督正常性建模方法（Generative Self-supervised Normality Modeling）

**核心定义**：通过 masked video modeling、视频补全、扩散去噪、帧/特征预测等自监督或生成式任务学习正常视频规律，并以重构偏差、预测偏差或分布偏离识别异常。

#### 1.4 知识蒸馏与轻量化部署方法（Knowledge Distillation and Efficient Adaptation）

**核心定义**：通过教师-学生蒸馏、参数高效微调、轻量检测头或特征压缩，将大型视觉基础模型的表征能力迁移到实时、低成本、可部署的 VAD 系统中。

### 2. VLM：视觉-语言开放词汇路线

#### 2.1 视觉-语言对齐适配方法（Vision-Language Alignment Adaptation）

**核心定义**：将视频视觉特征映射到文本语义空间，使异常检测从纯视觉异常分数判断扩展为视觉片段与异常语义描述之间的匹配问题。

#### 2.2 Prompt-guided 异常语义建模方法（Prompt-guided Anomaly Semantics Modeling）

**核心定义**：通过手工提示、可学习提示、正常/异常上下文提示或类别描述提示，为 VAD 注入语义先验，增强模型对异常类型和异常边界的识别能力。

#### 2.3 开放词汇异常识别方法（Open-vocabulary Anomaly Recognition）

**核心定义**：面向训练阶段未见过的异常类别，将 VAD 从封闭二分类扩展为类别无关检测与类别相关识别，使模型能够处理开放世界中的未知异常。

#### 2.4 文本辅助弱监督 VAD 方法（Text-assisted Weakly Supervised VAD）

**核心定义**：在视频级弱标签学习中引入文本语义约束、异常类别描述或视觉-语言相似度，以缓解弱监督标签噪声并提升异常片段定位质量。

### 3. MLLM：面向VAU的多模态大模型路线

#### 3.1 长视频输入与时序 Grounding 方法（Long-video Input and Temporal Grounding）

**核心定义**：通过关键帧采样、异常候选片段提取、长视频记忆压缩、时间戳建模或多粒度片段索引，使 MLLM 能在长视频中定位异常发生的时间范围。

#### 3.2 异常描述与视频问答方法（Anomaly Captioning and Video Question Answering）

**核心定义**：面向异常片段生成自然语言描述，并回答 what、where、when、who、how severe 等问题，使输出从异常分数升级为可读的语义解释。

#### 3.3 异常因果与后果推理方法（Causal and Consequence Reasoning）

**核心定义**：推断异常事件的触发原因、对象关系、发展过程、潜在后果和风险等级，是 VAU 区别于传统 VAD 的核心认知能力。

#### 3.4 可信异常理解方法（Trustworthy Video Anomaly Understanding）

**核心定义**：围绕幻觉抑制、证据对齐、解释忠实性、时序一致性和可验证推理构建约束机制，提升 MLLM 异常理解结果的可靠性与可部署性。

## 二、各子类匹配文献清单

### 1. VFM：视觉基础模型迁移路线

#### 1.1 视觉基础表征迁移方法

**核心定义**：利用大规模视觉或视频预训练模型作为特征提取底座，提升 VAD 在复杂场景、少样本和跨域条件下的视觉表征泛化能力。

**代表性文献（4篇）**

1. [VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_VideoMAE_V2_Scaling_Video_Masked_Autoencoders_With_Dual_Masking_CVPR_2023_paper.html)
   - 作者/单位：Limin Wang，南京大学
   - 发表信息：2023，CVPR，CCF A
   - 核心贡献：扩展视频 MAE 预训练规模，验证其作为通用视频表征底座的迁移能力。
   - 分类依据：核心贡献在大规模视频视觉表征迁移，而非语言语义或因果理解。

2. [InternVideo: General Video Foundation Models via Generative and Discriminative Learning](https://dblp.org/rec/conf/cvpr/WangLLZG0WZ023.html)
   - 作者/单位：Yi Wang，上海人工智能实验室
   - 发表信息：2023，CVPR，CCF A
   - 核心贡献：结合生成式与判别式学习构建通用视频基础模型。
   - 分类依据：作为视频基础模型底座，可支撑 VAD 特征迁移和跨任务泛化。

3. [ImageBind: One Embedding Space To Bind Them All](https://dblp.org/rec/conf/cvpr/GirdharELSAJM23)
   - 作者/单位：Rohit Girdhar，Meta AI
   - 发表信息：2023，CVPR，CCF A
   - 核心贡献：构建图像、文本、音频等多模态统一嵌入空间。
   - 分类依据：提供跨模态视觉基础表征，可作为多模态 VAD/VAU 的表征底座。

4. [Segment Anything](https://mlanthology.org/iccv/2023/kirillov2023iccv-segment/)
   - 作者/单位：Alexander Kirillov，Meta AI
   - 发表信息：2023，ICCV，CCF A
   - 核心贡献：提出 promptable 通用视觉分割基础模型和大规模分割数据。
   - 分类依据：可为对象级异常区域提取和空间定位提供视觉基础能力。

#### 1.2 时序定位增强方法

**核心定义**：在视觉基础模型之上引入时序 Transformer、时空注意力、长期上下文聚合或局部-全局动态建模，以增强异常片段的时间定位和动态变化感知能力。

**代表性文献（4篇）**

1. [Multi-Scale Video Anomaly Detection by Multi-Grained Spatio-Temporal Representation Learning](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Multi-Scale_Video_Anomaly_Detection_by_Multi-Grained_Spatio-Temporal_Representation_Learning_CVPR_2024_paper.html)
   - 作者/单位：Menghao Zhang，北京邮电大学
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：设计连续性判断、断点定位和缺失帧估计等 proxy tasks 学习多粒度时空表征。
   - 分类依据：核心目标是提升时空表征和小尺度异常定位能力。

2. [Learning Spatio-Temporal Relations with Multi-Scale Integrated Perception for Video Anomaly Detection](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=202402253151385444)
   - 作者/单位：Hongyu Ye，上海交通大学
   - 发表信息：2024，ICASSP，领域 B 类/信号处理主会
   - 核心贡献：通过多尺度 patch 关系和层次图卷积建模异常区域的时空关系。
   - 分类依据：强调多尺度空间关系与时序编码，属于时序定位增强路线。

3. [Long Short-Term Dynamic Prototype Alignment Learning for Video Anomaly Detection](https://www.ijcai.org/proceedings/2024/96)
   - 作者/单位：Chao Huang，论文单位以 IJCAI 论文首页为准
   - 发表信息：2024，IJCAI，CCF A
   - 核心贡献：引入长短期动态原型对齐机制，增强远距离帧预测和动态模式建模。
   - 分类依据：以长期动态原型和短期运动匹配改进时间依赖建模。

4. [Rethinking Prediction-Based Video Anomaly Detection from Local-Global Normality Perspective](https://www.sciencedirect.com/science/article/abs/pii/S0957417424024485)
   - 作者/单位：Mengyang Zhao，论文单位以期刊首页为准
   - 发表信息：2025，Expert Systems with Applications，领域代表期刊
   - 核心贡献：提出局部-全局正常性联合建模，解释预测式 VAD 在不同场景中的适应差异。
   - 分类依据：核心在局部与全局时序正常性的互补建模。

#### 1.3 生成式自监督正常性建模方法

**核心定义**：通过 masked video modeling、视频补全、扩散去噪、帧/特征预测等自监督或生成式任务学习正常视频规律，并以重构偏差、预测偏差或分布偏离识别异常。

**代表性文献（4篇）**

1. [Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors](https://openaccess.thecvf.com/content/CVPR2024/html/Ristea_Self-Distilled_Masked_Auto-Encoders_are_Efficient_Video_Anomaly_Detectors_CVPR_2024_paper.html)
   - 作者/单位：Nicolae-Cătălin Ristea，University of Bucharest
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：用轻量 masked AE、自蒸馏解码器差异和合成异常增强实现高效 VAD。
   - 分类依据：以 masked reconstruction 和自监督正常性恢复作为异常检测核心。

2. [Detecting Anomalous Events from Unlabeled Videos via Temporal Masked Auto-Encoding](https://ai.updf.com/paper-detail/detecting-anomalous-events-from-unlabeled-videos-via-temporal-masked-auto-hu-yu-d1000a75f3a6bf7c71b2d30f0a9d74c541977d4d)
   - 作者/单位：Jingtao Hu，论文单位以 ICME 论文首页为准
   - 发表信息：2022，ICME，CCF B
   - 核心贡献：提出 temporal masked auto-encoding，通过时序掩码恢复判断异常。
   - 分类依据：属于典型 masked self-supervised normality modeling。

3. [Diffusion-based Normality Pre-training for Weakly Supervised Video Anomaly Detection](https://www.sciencedirect.com/science/article/pii/S0957417424008790)
   - 作者/单位：Suvramalya Basak，Indian Institute of Information Technology Allahabad
   - 发表信息：2024，Expert Systems with Applications，领域代表期刊
   - 核心贡献：利用扩散式正常性预训练学习正常特征分布，再用于弱监督异常定位。
   - 分类依据：以 diffusion normality pre-training 作为核心机制。

4. [Generative Cooperative Learning for Unsupervised Video Anomaly Detection](https://dblp.org/rec/conf/cvpr/ZaheerMKSY22.html)
   - 作者/单位：M. Zaigham Zaheer，Mohamed bin Zayed University of Artificial Intelligence
   - 发表信息：2022，CVPR，CCF A
   - 核心贡献：用生成式协同学习建模正常模式并检测异常偏离。
   - 分类依据：属于生成式正常性建模，补充扩散模型之前的生成式路线。

#### 1.4 知识蒸馏与轻量化部署方法

**核心定义**：通过教师-学生蒸馏、参数高效微调、轻量检测头或特征压缩，将大型视觉基础模型的表征能力迁移到实时、低成本、可部署的 VAD 系统中。

**代表性文献（4篇）**

1. [Lightning Fast Video Anomaly Detection via Multi-scale Adversarial Distillation](https://www.sciencedirect.com/science/article/pii/S1077314224001553)
   - 作者/单位：Florinel-Alin Croitoru，University of Bucharest
   - 发表信息：2024，CVIU，领域代表期刊
   - 核心贡献：用多教师模型和对抗蒸馏获得极高速帧级 VAD。
   - 分类依据：核心机制是 teacher-student distillation 与速度-精度折中。

2. [Real-Time Weakly Supervised Video Anomaly Detection](https://openaccess.thecvf.com/content/WACV2024/html/Karim_Real-Time_Weakly_Supervised_Video_Anomaly_Detection_WACV_2024_paper.html)
   - 作者/单位：Hamza Karim，University of Nevada, Reno
   - 发表信息：2024，WACV，领域代表会议
   - 核心贡献：提出端到端实时弱监督 VAD，显著降低异常决策延迟。
   - 分类依据：目标直接面向实时部署和低延迟检测。

3. [Ensemble-Based Knowledge Distillation for Video Anomaly Detection](https://www.mdpi.com/2076-3417/14/3/1032)
   - 作者/单位：Burçak Asal，Hacettepe University
   - 发表信息：2024，Applied Sciences，领域代表期刊
   - 核心贡献：利用多教师蒸馏缓解跨数据集适应中的灾难性遗忘。
   - 分类依据：以 ensemble knowledge distillation 支撑 VAD 适配与迁移。

4. [STNMamba: Mamba-based Spatial-Temporal Normality Learning for Video Anomaly Detection](https://www.emergentmind.com/papers/2412.20084)
   - 作者/单位：Zhangxun Li，论文单位以 arXiv/PDF 首页为准
   - 发表信息：2024，arXiv，预印本
   - 核心贡献：用 Mamba 模块降低长程时空正常性建模的计算负担。
   - 分类依据：虽非蒸馏方法，但代表轻量化时空建模和高效部署方向。

### 2. VLM：视觉-语言开放词汇路线

#### 2.1 视觉-语言对齐适配方法

**核心定义**：将视频视觉特征映射到文本语义空间，使异常检测从纯视觉异常分数判断扩展为视觉片段与异常语义描述之间的匹配问题。

**代表性文献（4篇）**

1. [VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection](https://ojs.aaai.org/index.php/AAAI/article/view/28423)
   - 作者/单位：Peng Wu，西北工业大学
   - 发表信息：2024，AAAI，CCF A
   - 核心贡献：利用冻结 CLIP 与双分支结构实现粗粒度和细粒度弱监督 VAD。
   - 分类依据：核心是将 CLIP 的视觉-语言对齐能力适配到 VAD。

2. [Harnessing Large Language Models for Training-free Video Anomaly Detection](https://cvpr.thecvf.com/virtual/2024/poster/29246)
   - 作者/单位：Luca Zanella，University of Trento
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：提出 LAVAD，用 VLM 生成描述并用 LLM 聚合异常分数。
   - 分类依据：以视觉-语言描述作为异常评分中介，属于语言对齐适配。

3. [Unsupervised Video Anomaly Detection Based on Similarity with Predefined Text Descriptions](https://pure.korea.ac.kr/en/publications/unsupervised-video-anomaly-detection-based-on-similarity-with-pre/)
   - 作者/单位：Jaehyun Kim，Korea University
   - 发表信息：2023，Sensors，领域代表期刊
   - 核心贡献：用预定义文本描述与视频帧 CLIP 相似度进行无监督 VAD。
   - 分类依据：将异常判断转化为视频-文本相似度匹配。

4. [VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models](https://vera-framework.github.io/)
   - 作者/单位：Muchao Ye，The University of Iowa
   - 发表信息：2025，CVPR，CCF A
   - 核心贡献：通过可学习引导问题让冻结 VLM 同时进行检测和解释。
   - 分类依据：核心是 verbalized VLM adaptation，用语言化问题引导异常识别。

#### 2.2 Prompt-guided 异常语义建模方法

**核心定义**：通过手工提示、可学习提示、正常/异常上下文提示或类别描述提示，为 VAD 注入语义先验，增强模型对异常类型和异常边界的识别能力。

**代表性文献（4篇）**

1. [Prompt-Enhanced Multiple Instance Learning for Weakly Supervised Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Prompt-Enhanced_Multiple_Instance_Learning_for_Weakly_Supervised_Video_Anomaly_Detection_CVPR_2024_paper.html)
   - 作者/单位：Junxi Chen，论文单位以 CVPR 论文首页为准
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：设计 abnormal-aware prompts 与 normal context prompts，增强 MIL 异常模式表达。
   - 分类依据：核心机制是 prompt-enhanced anomaly semantics。

2. [Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Text_Prompt_with_Normality_Guidance_for_Weakly_Supervised_Video_Anomaly_CVPR_2024_paper.html)
   - 作者/单位：Zhiwei Yang，论文单位以 CVPR 论文首页为准
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：利用文本 prompt 与正常性视觉 prompt 生成更可靠伪标签。
   - 分类依据：以 normality-guided text prompt 改善异常语义匹配。

3. [Learning Prompt-Enhanced Context Features for Weakly-Supervised Video Anomaly Detection](https://pubmed.ncbi.nlm.nih.gov/39236124/)
   - 作者/单位：Yujiang Pu，论文单位以 TIP 论文首页为准
   - 发表信息：2024，IEEE TIP，CCF A
   - 核心贡献：学习 prompt-enhanced context features 以提升弱监督异常定位。
   - 分类依据：通过 prompt 语义上下文增强片段级异常判别。

4. [VPE-WSVAD: Visual Prompt Exemplars for Weakly-supervised Video Anomaly Detection](https://www.sciencedirect.com/science/article/abs/pii/S0950705124006129)
   - 作者/单位：Yong Su，论文单位以期刊首页为准
   - 发表信息：2024，Knowledge-Based Systems，领域代表期刊
   - 核心贡献：用视觉 prompt exemplars 提升弱监督异常与正常区分。
   - 分类依据：属于 prompt-guided 语义建模的视觉提示分支。

#### 2.3 开放词汇异常识别方法

**核心定义**：面向训练阶段未见过的异常类别，将 VAD 从封闭二分类扩展为类别无关检测与类别相关识别，使模型能够处理开放世界中的未知异常。

**代表性文献（4篇）**

1. [Open-Vocabulary Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Open-Vocabulary_Video_Anomaly_Detection_CVPR_2024_paper.html)
   - 作者/单位：Peng Wu，西北工业大学
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：将 OVVAD 拆分为类别无关检测和类别相关分类，并引入语义知识与异常合成。
   - 分类依据：直接定义并解决 open-vocabulary VAD。

2. [Multimodal Evidential Learning for Open-World Weakly-Supervised Video Anomaly Detection](https://dblp.org/rec/journals/tmm/HuangHJWWZ25)
   - 作者/单位：Chao Huang，论文单位以 TMM 论文首页为准
   - 发表信息：2025，IEEE TMM，CCF B
   - 核心贡献：用视觉-语言关联和证据学习处理 seen/unseen 异常。
   - 分类依据：面向 open-world WSVAD，强调未知异常识别。

3. [Towards Open Set Video Anomaly Detection](https://dblp.org/rec/conf/eccv/ZhuBY22)
   - 作者/单位：Yuansheng Zhu，论文单位以 ECCV 论文首页为准
   - 发表信息：2022，ECCV Workshop，开放集基准代表
   - 核心贡献：较早讨论开放集 VAD 设定和未知异常检测问题。
   - 分类依据：作为开放词汇/开放世界 VAD 的问题前置与基准支撑。

4. [UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Acsintoae_UBnormal_New_Benchmark_for_Supervised_Open-Set_Video_Anomaly_Detection_CVPR_2022_paper.html)
   - 作者/单位：Andra Acsintoae，University of Bucharest
   - 发表信息：2022，CVPR，CCF A
   - 核心贡献：构建监督开放集 VAD 基准，区分训练异常与测试未知异常。
   - 分类依据：为开放类别异常泛化提供数据和评价基础。

#### 2.4 文本辅助弱监督 VAD 方法

**核心定义**：在视频级弱标签学习中引入文本语义约束、异常类别描述或视觉-语言相似度，以缓解弱监督标签噪声并提升异常片段定位质量。

**代表性文献（4篇）**

1. [TDSD: Text-Driven Scene-Decoupled Weakly Supervised Video Anomaly Detection](https://openreview.net/forum?id=TAVtkpjS9P)
   - 作者/单位：Shengyang Sun，论文单位以 ACM MM 论文首页为准
   - 发表信息：2024，ACM MM，CCF A
   - 核心贡献：用文本驱动场景解耦处理 scene-dependent 弱监督 VAD。
   - 分类依据：文本语义用于解耦场景和异常，是文本辅助弱监督代表。

2. [CMHKF: Cross-Modality Heterogeneous Knowledge Fusion for Weakly Supervised Video Anomaly Detection](https://aclanthology.org/2025.acl-long.1524/)
   - 作者/单位：Guohua Wang，论文单位以 ACL 论文首页为准
   - 发表信息：2025，ACL，CCF A
   - 核心贡献：融合视频、音频和文本异构知识以改善弱监督检测。
   - 分类依据：以跨模态文本知识辅助弱监督异常定位。

3. [BatchNorm-Based Weakly Supervised Video Anomaly Detection](https://eurekamag.com/research/099/893/099893380.php)
   - 作者/单位：Yixuan Zhou，论文单位以 TCSVT 论文首页为准
   - 发表信息：2024，IEEE TCSVT，CCF B
   - 核心贡献：利用 BatchNorm 特征偏离作为弱监督异常判据。
   - 分类依据：虽非纯文本方法，但可作为文本辅助弱监督路线的视觉统计对照基线。

4. [WSVAD-CLIP: Temporally Aware and Prompt Learning with CLIP for Weakly Supervised Video Anomaly Detection](https://pubmed.ncbi.nlm.nih.gov/41150030/)
   - 作者/单位：Min Li，North China University of Technology
   - 发表信息：2025，Journal of Imaging，领域代表期刊
   - 核心贡献：结合 CLIP、时序感知与 prompt learning 进行弱监督 VAD。
   - 分类依据：以 CLIP 文本提示增强弱监督时序异常定位。

### 3. MLLM：面向VAU的多模态大模型路线

#### 3.1 长视频输入与时序 Grounding 方法

**核心定义**：通过关键帧采样、异常候选片段提取、长视频记忆压缩、时间戳建模或多粒度片段索引，使 MLLM 能在长视频中定位异常发生的时间范围。

**代表性文献（4篇）**

1. [MovieChat: From Dense Token to Sparse Memory for Long Video Understanding](https://openaccess.thecvf.com/content/CVPR2024/html/Song_MovieChat_From_Dense_Token_to_Sparse_Memory_for_Long_Video_CVPR_2024_paper.html)
   - 作者/单位：Enxin Song，Zhejiang University
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：用稀疏记忆机制处理长视频理解和问答。
   - 分类依据：解决长视频输入压缩和长程上下文保持问题。

2. [VTimeLLM: Empower LLM to Grasp Video Moments](https://colab.ws/articles/10.1109%2Fcvpr52733.2024.01353)
   - 作者/单位：Bin Huang，Tsinghua University
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：使 LLM 能够进行视频 moment 定位和时间边界理解。
   - 分类依据：直接支撑 VAU 中异常片段 temporal grounding。

3. [TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding](https://huggingface.co/ShuhuaiRen/TimeChat-7b)
   - 作者/单位：Shuhuai Ren，论文单位以 arXiv/PDF 首页为准
   - 发表信息：2023，arXiv，预印本
   - 核心贡献：构建时间敏感视频指令数据和长视频 MLLM。
   - 分类依据：核心是时间戳感知和长视频时序理解。

4. [MVBench: A Comprehensive Multi-modal Video Understanding Benchmark](https://openaccess.thecvf.com/content/CVPR2024/html/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.html)
   - 作者/单位：Kunchang Li，论文单位以 CVPR 论文首页为准
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：构建覆盖 20 类动态视频任务的 MLLM 时序理解基准。
   - 分类依据：为长视频时序 grounding 和动态理解提供系统评估。

#### 3.2 异常描述与视频问答方法

**核心定义**：面向异常片段生成自然语言描述，并回答 what、where、when、who、how severe 等问题，使输出从异常分数升级为可读的语义解释。

**代表性文献（4篇）**

1. [Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models](https://aclanthology.org/2024.acl-long.679/)
   - 作者/单位：Muhammad Maaz，Mohamed bin Zayed University of Artificial Intelligence
   - 发表信息：2024，ACL，CCF A
   - 核心贡献：构建视频指令数据和视频对话模型，实现细粒度视频问答。
   - 分类依据：为异常问答和自然语言解释提供通用 MLLM 框架。

2. [Video-LLaVA: Learning United Visual Representation by Alignment Before Projection](https://aclanthology.org/2024.emnlp-main.342/)
   - 作者/单位：Bin Lin，Nanjing University
   - 发表信息：2024，EMNLP，CCF B/ACL 体系主会
   - 核心贡献：统一图像与视频表征，使 LLM 支持视频对话与问答。
   - 分类依据：面向视频自然语言交互，支撑异常描述与问答。

3. [Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding](https://aclanthology.org/2023.emnlp-demo.49/)
   - 作者/单位：Hang Zhang，DAMO Academy
   - 发表信息：2023，EMNLP Demo，CCF B/ACL 体系
   - 核心贡献：引入视觉与音频编码器，构建指令微调视频理解模型。
   - 分类依据：强调视频问答和多模态对话，可迁移到异常解释。

4. [Video Anomaly Detection and Explanation via Large Language Models](https://dblp.org/rec/journals/corr/abs-2401-05702.html)
   - 作者/单位：Hui Lv，论文单位以 arXiv/PDF 首页为准
   - 发表信息：2024，arXiv，预印本
   - 核心贡献：探索 VLLM 在 VAD 中同时完成检测与文本解释。
   - 分类依据：直接面向异常解释输出，是 VAU 描述方向的早期尝试。

#### 3.3 异常因果与后果推理方法

**核心定义**：推断异常事件的触发原因、对象关系、发展过程、潜在后果和风险等级，是 VAU 区别于传统 VAD 的核心认知能力。

**代表性文献（4篇）**

1. [Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly](https://colab.ws/articles/10.1109%2Fcvpr52733.2024.01778)
   - 作者/单位：Hang Du，Beijing University of Posts and Telecommunications
   - 发表信息：2024，CVPR，CCF A
   - 核心贡献：提出 CUVA，标注异常类型、时间、描述、原因和后果，并提出理解评价。
   - 分类依据：直接定义 VAU 中 what/why/how 因果理解任务。

2. [NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions](https://dblp.org/rec/conf/cvpr/XiaoSGSZS21.html)
   - 作者/单位：Junbin Xiao，论文单位以 CVPR 论文首页为准
   - 发表信息：2021，CVPR，CCF A
   - 核心贡献：面向视频中的 temporal action explanation 构建因果/解释型问答。
   - 分类依据：为异常因果问答提供通用视频解释型 QA 基础。

3. [STAR: A Benchmark for Situated Reasoning in Real-World Videos](https://dblp.org/rec/conf/nips/WuCSKK21.html)
   - 作者/单位：Bo Wu，论文单位以 NeurIPS 论文首页为准
   - 发表信息：2021，NeurIPS，CCF A
   - 核心贡献：构建真实视频情境推理 benchmark，覆盖交互、时序和因果关系。
   - 分类依据：支撑 VAU 中对象关系和事件发展推理。

4. [Video-and-Language Event Prediction: A Baseline for Predicting Human Actions from Video](https://dblp.org/rec/conf/emnlp/LeiB20.html)
   - 作者/单位：Jie Lei，论文单位以 EMNLP 论文首页为准
   - 发表信息：2020，EMNLP，CCF B/ACL 体系主会
   - 核心贡献：提出基于视频与语言的未来事件预测任务。
   - 分类依据：虽早于主体时间范围，但可作为后果推理与事件预测的奠基参考。

#### 3.4 可信异常理解方法

**核心定义**：围绕幻觉抑制、证据对齐、解释忠实性、时序一致性和可验证推理构建约束机制，提升 MLLM 异常理解结果的可靠性与可部署性。

**代表性文献（4篇）**

1. [Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](https://dblp.org/rec/conf/cvpr/FuDLLRZWZSZCLLZ25)
   - 作者/单位：Chaoyou Fu，论文单位以 CVPR 论文首页为准
   - 发表信息：2025，CVPR，CCF A
   - 核心贡献：构建覆盖短中长视频、多模态输入和多领域任务的 MLLM 评测基准。
   - 分类依据：为 VAU 系统可靠性和长视频能力评估提供基础。

2. [TemporalBench: Benchmarking Fine-grained Temporal Understanding for Multimodal Video Models](https://huggingface.co/papers/2410.10818)
   - 作者/单位：Mu Cai，Microsoft Research
   - 发表信息：2024，arXiv，预印本
   - 核心贡献：评估动作频率、运动幅度、事件顺序等细粒度时间理解能力。
   - 分类依据：用于检验 VAU 解释是否具备时序一致性。

3. [VidHalluc: Evaluating Temporal Hallucinations in Multimodal Large Language Models for Video Understanding](https://huggingface.co/papers/2412.03735)
   - 作者/单位：Chaoyu Li，论文单位以 arXiv/PDF 首页为准
   - 发表信息：2024，arXiv，预印本
   - 核心贡献：构建视频 MLLM 时间幻觉评测并提出缓解思路。
   - 分类依据：直接面向视频理解幻觉，是可信 VAU 的关键支撑。

4. [How Good is my Video-LMM? Complex Video Reasoning and Robustness Evaluation Suite for Video-LMMs](https://cvpr.thecvf.com/virtual/2025/35643)
   - 作者/单位：Muhammad Uzair Khattak，论文单位以 CVPR Workshop 论文首页为准
   - 发表信息：2025，CVPR Workshop，领域代表
   - 核心贡献：从复杂视频推理和鲁棒性角度评估 Video-LMM。
   - 分类依据：支撑异常理解系统的鲁棒性、提示敏感性和可靠性分析。

## 三、2025–2026 高水平文献补强清单

> 说明：以下条目用于补强近两年顶会/高水平会议覆盖，可在正式正文中按“唯一归属”并入对应二级子类。优先选取与 VAD/VAU 直接相关的 CVPR/ICCV/AAAI/ACM ICMR 论文。

| 建议归入子类                    | 完整论文标题                                                                                                                                                                                                                                                                                | 第一作者+所属单位                                                     | 发表信息                | 核心技术贡献                                         | 归类依据                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------- | ---------------------------------------------- | ------------------------------------------------------- |
| 1.2 时序定位增强方法              | [Track Any Anomalous Object: A Granular Video Anomaly Detection Pipeline](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_Track_Any_Anomalous_ObjectA_Granular_Video_Anomaly_Detection_Pipeline_CVPR_2025_paper.html)                                                       | Yuzhi Huang，论文单位以 CVPR 论文首页为准                                 | 2025，CVPR，CCF A     | 将异常检测扩展到对象/像素级跟踪，实现更细粒度定位。                     | 重点在 granular localization 与异常对象追踪，补强 VFM 时序定位分支。        |
| 1.3 生成式自监督正常性建模方法         | [Video Anomaly Detection with Motion and Appearance Guided Patch Diffusion Model](https://ojs.aaai.org/index.php/AAAI/article/view/33169)                                                                                                                                             | Hang Zhou，Huazhong University of Science and Technology       | 2025，AAAI，CCF A     | 以运动和外观条件引导 patch diffusion，建模正常模式偏离。           | 属于扩散生成式正常性建模，直接补足 diffusion-VAD 高水平工作。                  |
| 2.1 视觉-语言对齐适配方法           | [Retrieval-Guided Contextual Inference for Training-Free Video Anomaly Detection in Low-Light Scenarios](https://doi.org/10.1145/3805622.3810823)                                                                                                                                     | Mengjingcheng Mo，论文单位以 ACM 论文首页为准                             | 2026，ACM ICMR，CCF B | 通过语义上下文与检索上下文稳定低光训练自由 VAD 推理。                  | 以 VLM/LLM 上下文推理完成 training-free VAD，强化鲁棒开放场景。           |
| 2.3 开放词汇异常识别方法            | [Anomize: Better Open Vocabulary Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Anomize_Better_Open_Vocabulary_Video_Anomaly_Detection_CVPR_2025_paper.html)                                                                                         | Fei Li，论文单位以 CVPR 论文首页为准                                      | 2025，CVPR，CCF A     | 结合多层视觉信息、文本匹配和标签关系，缓解 novel anomaly 检测歧义。      | 直接面向 OVVAD，是开放词汇异常识别的 2025 核心进展。                        |
| 2.4 文本辅助弱监督 VAD 方法        | [Learning from Noisy Supervision: A Denoising-Debiasing Framework for Weakly Supervised Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_Learning_from_Noisy_Supervision_A_Denoising-Debiasing_Framework_for_Weakly_Supervised_CVPR_2026_paper.html) | Yaxin Zhao，Nankai University                                  | 2026，CVPR，CCF A     | 提出 D²MIL，用动态去噪与 VLM 去偏缓解弱监督 MIL 噪声。            | 弱监督主线结合 VLM 重评估难例，适合放入文本辅助弱监督分支。                        |
| 2.4 文本辅助弱监督 VAD 方法        | [Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Weakly_Supervised_Video_Anomaly_Detection_with_Anomaly-Connected_Components_and_Intention_CVPR_2026_paper.html)        | Yu Wang，论文单位以 CVPR 论文首页为准                                     | 2026，CVPR，CCF A     | 通过异常连通组件、意图感知和属性信息学习异常语义。                      | 强调 anomaly semantics 与 intention reasoning，是弱监督语义化趋势代表。 |
| 3.1 长视频输入与时序 Grounding 方法 | [Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity](https://dblp.org/rec/conf/cvpr/ZhangX0ZHGZ0S25.html)                                                                                                                                                   | Huaxin Zhang，Huazhong University of Science and Technology    | 2025，CVPR，CCF A     | 构建 HIVAU-70k 多粒度 VAU 数据，并用异常聚焦采样处理长视频。         | 直接面向长视频、多粒度异常理解和 temporal grounding。                    |
| 3.2 异常描述与视频问答方法           | [Aligning Effective Tokens with Video Anomaly in Large Language Models](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Aligning_Effective_Tokens_with_Video_Anomaly_in_Large_Language_Models_ICCV_2025_paper.html)                                                          | Yingxian Chen，The University of Hong Kong                     | 2025，ICCV，CCF A     | 提出 VA-GPT，通过时空有效 token 对齐完成异常总结和定位。            | 面向异常感知 MLLM，可支撑异常描述、问答与定位解释。                            |
| 3.2 异常描述与视频问答方法           | [No Need For Real Anomaly: MLLM Empowered Zero-Shot Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2026/html/Dai_No_Need_For_Real_Anomaly_MLLM_Empowered_Zero-Shot_Video_Anomaly_CVPR_2026_paper.html)                                                            | Zunkai Dai，Beijing University of Posts and Telecommunications | 2026，CVPR，CCF A     | 提出 LAVIDA，用伪异常暴露、MLLM 语义理解和 token 压缩实现零样本 VAD。 | 将 MLLM 用于零样本异常检测和语义理解，可作为 VAD→VAU 过渡代表。                 |

