### 1 引言

#### 1.1 VAD 研究背景与应用价值

视频异常检测（Video Anomaly Detection, VAD）面向智能监控、公共安全、交通管理、工业巡检、校园与公共空间安防等场景，目标是在长时视频中识别低频、开放类别、场景依赖强且标注代价高的异常事件。早期 VAD 主要回答“是否异常、何时异常、何处异常”，核心输出通常是视频级标签、帧级/片段级异常分数、时间定位或空间定位。

#### 1.2 VAD 与 VAU 的任务边界

VAD 是感知层任务，关注 anomaly detection/localization：给定视频序列，输出异常分数、时间边界、空间区域或类别判断。VAU（Video Anomaly Understanding）是认知层任务，进一步关注 anomaly description/reasoning/explanation：异常是什么、为何发生、造成何种后果、严重程度如何、能否形成可解释的因果链。二者不是替代关系，而是层级演进关系：VAD 为 VAU 提供候选异常片段与定位基础，VAU 在此基础上进行语义、因果和决策层理解。

#### 1.3 现有综述的不足

已有 VAD 综述多按监督范式、模型结构、数据集和评价指标展开，对大模型时代的 VFM、VLM、MLLM 路线缺少统一归纳；同时多数综述仍以 detection 为中心，对 anomaly understanding、causation understanding、open-vocabulary anomaly recognition、text-grounded anomaly explanation 等趋势讨论不足。

#### 1.4 本文贡献与组织结构

本文以“双主线”组织 VAD/VAU 研究：一条主线系统梳理传统深度学习 VAD，包括无监督、弱监督、全监督/开放集监督方法；另一条主线梳理大模型驱动的 VAD/VAU，包括 VFM、VLM 与 MLLM。全文建议结构为：第 2 章定义任务、数据集与指标；第 3 章梳理传统深度学习 VAD；第 4 章讨论大模型驱动的 VAD/VAU；第 5 章进行传统方法与大模型方法对比；第 6 章总结关键不足；第 7 章展望未来方向；第 8 章总结全文。

### 2 基于传统深度学习的视频异常检测方法



#### 2.1 无监督 VAD：仅正常样本训练

无监督 VAD 的核心假设是正常视频可被模型稳定重构、预测或聚类到紧致正常流形，而异常事件会偏离该流形。其优势是无需异常标注，适合异常稀缺的真实场景；局限在于 normality 假设强、跨场景泛化弱，且强模型可能同时重构异常。

##### 2.1.1 基于重构的方法

核心逻辑是模型只学习正常样本的外观、运动或对象级模式，异常样本因偏离正常分布产生较高重构误差。早期 sparse coding、局部统计聚合与浅层特征建立了“正常模式可重构、异常难重构”的基本范式；进入深度学习阶段后，Conv-AE、3D Conv-AE、LSTM-AE 与 stacked denoising AE 被用于学习 appearance-motion 表征。中期方法将空间纹理与时间动态联合建模，并从整帧重构转向对象级、轨迹级和外观-运动一致性建模。近年方法进一步引入多尺度重构、残差/注意力 AE、masked autoencoding 与对象关系恢复，但过强重构能力仍可能削弱异常与正常的误差差异。

##### 2.1.2 基于时序预测的方法

核心逻辑是正常事件具有可预测的时空演化规律，异常会造成未来帧、光流、轨迹或骨架运动预测误差升高。ConvLSTM 与未来帧预测将 VAD 从静态重构推进到动态预测，Liu et al. (2018) 明确把 future frame prediction 建立为 VAD 新基线。后续方法引入光流预测、双向预测、帧插值、appearance-motion 联合预测和骨架轨迹预测，以增强对运动异常、结构化人体行为异常和短时突变的敏感性。近年 Transformer、图卷积、masked temporal modeling 与 local-global normality 建模进一步捕获长短期动态，但该类方法仍易受遮挡、相机抖动、复杂背景运动和正常行为多样性影响。

##### 2.1.3 基于度量、原型与记忆的方法

核心逻辑是正常样本在特征空间中形成紧致流形或多个正常原型，异常样本表现为远离正常簇、低相似度或低正常性分数。早期 deep feature + one-class classifier 形成“正常边界”判别思路；memory-augmented autoencoder 与 memory-guided normality 用记忆项存储正常模式，限制 AE 对异常的泛化重构能力；feature compactness/separateness loss 进一步强化正常特征紧致性。近年 pose graph clustering、prototype alignment、appearance-motion prototype、normality network 等方法将原型扩展到姿态图、对象轨迹、外观-运动联合空间。该分支比单纯重构误差更利于解释“偏离何种正常模式”，但仍依赖训练场景中的正常模式覆盖。

##### 2.1.4 生成对抗类无监督 VAD

核心逻辑是 GAN 或扩散模型学习正常视频分布，通过生成质量、判别器响应、预测残差或分布偏离度识别异常。GAN-based abnormal event detection 将生成器用于重构/预测正常模式，将判别器用于区分真实正常与异常偏离。后续 dual-discriminator GAN、cross-channel adversarial learning、GAN-based future frame prediction 结合 RGB、光流和上下文一致性，改善单一生成误差不稳定的问题。近年 diffusion-based normality pre-training、feature prediction diffusion 与 denoising diffusion-augmented reconstruction 将生成式建模从 GAN 扩展到扩散模型，但训练稳定性、推理成本和异常可解释性仍是瓶颈。

#### 2.2 弱监督 VAD：视频级粗标签与 MIL

弱监督 VAD 使用视频级标签训练模型，在未裁剪长视频中定位异常片段，主流范式是 multiple instance learning (MIL)：视频为 bag，片段为 instance，异常视频至少包含一个异常片段，正常视频所有片段应为正常。

##### 2.2.1 基础 MIL 弱监督框架

Sultani et al. (2018) 基于 UCF-Crime 提出 ranking loss 与 MIL 框架，推动 VAD 从小规模无监督正常训练转向大规模真实监控弱监督学习。后续中心引导判别学习、normalcy suppression、feature magnitude learning、temporal relation learning 等方法围绕 top-k 片段选择、正常/异常间隔扩大和时间关系建模展开。近年 unbiased MIL、normality-guided MIL、batchnorm-based WS-VAD 与 real-time WS-VAD 关注偏置、效率和部署问题。该分支的根本难点是视频级标签到帧级边界之间存在噪声传播。

##### 2.2.2 时序注意力与伪标签优化弱监督

该分支利用 self-training、pseudo label、temporal attention、Transformer 或 graph label noise cleaning，逐步从粗标签中挖掘更可靠的异常片段。MIST 通过多实例自训练优化伪标签；后续工作引入 transformer-enabled temporal relation、self-guided temporal discriminative transformer、completeness and uncertainty of pseudo labels 等机制，以提升边界定位与长程依赖建模能力。其局限是伪标签质量高度依赖初始模型，异常边界模糊时易形成确认偏差；注意力权重不等价于因果解释，仍主要服务 VAD 定位。

##### 2.2.3 多模态弱监督

真实异常往往伴随声音、视觉和上下文线索，多模态弱监督通过音频-视觉融合提升暴力、碰撞、爆炸、尖叫等事件检测。XD-Violence 将 audio-visual violence detection 推向大规模弱监督；modality-aware contrastive instance learning、自蒸馏、多模态注意力融合等方法进一步解决模态异步、噪声和互补性问题。局限在于音频缺失、背景噪声、模态不同步与跨场景录音差异会显著影响鲁棒性；多数方法仍输出异常分数，而不是稳定的语义因果解释。

#### 2.3 全监督/开放集监督 VAD

全监督 VAD 使用帧级、片段级或空间级异常标注训练检测器，理论上能够获得更精确的异常边界，但标注成本高，且真实异常类别开放、长尾明显。开放集监督进一步要求模型在训练异常类别与测试异常类别不一致时仍能检测未知异常。UBnormal 通过合成/虚拟场景构建 supervised open-set VAD 基准，推动对开放类别泛化的系统评价。DoTA、Street Scene 与 Open-Vocabulary VAD 则分别从交通场景时空类别标注、实际检测协议和开放词汇类别识别等方向拓展了开放世界 VAD 的评价边界。

#### 2.4 传统深度学习方法的共性缺陷

1. **normality 假设依赖强**：正常行为在真实场景中高度多样，单一正常流形难以覆盖所有合法模式。
2. **语义表达能力弱**：传统 VAD 多输出异常分数、热力图或边界，难以回答“异常是什么、为什么异常”。
3. **开放世界泛化不足**：闭集类别、固定场景和固定摄像头训练限制了跨场景迁移。
4. **弱监督噪声难消除**：视频级标签到帧级定位之间存在天然不确定性。
5. **评价体系偏检测**：AUC、AP 与 EER 难以衡量解释、因果和后果推理质量。
6. **从 VAD 到 VAU 的任务断层**：传统方法解决“看见异常”，但很少形成“理解异常”的语义因果链。

### 3 基于大模型的 VAD 与 VAU 方法

#### 3.1 VFM：视觉基础模型迁移到 VAD

VFM 路线以 ViT、MAE、VideoMAE、InternVideo、SAM、DINO/自蒸馏等视觉基础模型为底座，重点提升跨场景视觉表征、少样本适应和时空特征质量。该类方法仍主要服务检测、评分与定位，通常不直接具备自然语言解释和因果推理能力。

#### 3.2 VLM：视觉-语言模型驱动的开放词汇 VAD

VLM 路线以 CLIP、video-language alignment、prompt learning 和 open-vocabulary recognition 为核心，将异常检测从封闭二分类推进到语义类别识别。VadCLIP、Open-Vocabulary VAD、Prompt-Enhanced MIL 与 LAVAD 等工作说明，文本提示和视觉语言对齐可使模型从“是否异常”部分走向“异常属于什么语义类别”。

#### 3.3 MLLM：面向 VAU 的异常描述、问答与因果理解

MLLM 路线将异常研究扩展到视频问答、事件描述、因果解释、严重程度判断与后果推理。CUVA 以 what/why/how 标注和 MMEval 推动 VAU benchmark 化，Holmes-VAU 进一步面向长视频、多粒度异常理解。该方向是 VAD 到 VAU 的关键跃迁，但仍面临长视频 temporal grounding、幻觉、因果忠实性、实时部署和评价标准不成熟等挑战。




| 方法类别                    | 完整论文标题                                                 | 发表年份 | 会议/期刊&CCF等级    | 核心贡献                                              | 引用用途                             | 方法概述                                                     | 优势                     | 局限性                     |
| --------------------------- | ------------------------------------------------------------ | -------: | -------------------- | ----------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ | ------------------------ | -------------------------- |
| 无监督/早期正常性建模       | Anomaly Detection in Crowded Scenes                          |     2010 | CVPR，CCF A          | 提出 UCSD Ped1/Ped2 与拥挤场景异常检测基准            | 奠定固定摄像头 VAD 数据与评价基础    | 建模拥挤场景正常动态并检测局部异常                           | 早期标准基准，便于比较   | 场景单一、规模较小         |
| 无监督/稀疏重构             | Abnormal Event Detection at 150 FPS in MATLAB                |     2013 | ICCV，CCF A          | 提出 Avenue 数据集并展示高速异常检测                  | 说明早期稀疏重构与小规模行为异常基准 | 用稀疏组合和局部异常得分定位异常                             | 高效、可解释             | 深度表征能力有限           |
| 无监督/深度重构             | Learning Temporal Regularity in Video Sequences              |     2016 | CVPR，CCF A          | 用卷积 AE 学习视频时序规律                            | 深度重构式 VAD 代表                  | 训练 AE 重构正常视频，异常以高误差显现                       | 无需异常标注             | 强 AE 可能重构异常         |
| 无监督/外观运动表征         | Detecting Anomalous Events in Videos by Learning Deep Representations of Appearance and Motion |     2017 | CVIU，CCF B 期刊     | 学习外观与运动深度表征并结合 one-class 判别           | 连接重构表征与度量判别               | stacked denoising AE 提取外观/运动特征                       | 特征表达强于手工特征     | 对场景迁移敏感             |
| 无监督/时空 AE              | Abnormal Event Detection in Videos Using Spatiotemporal Autoencoder |     2017 | 会议论文             | 使用时空 AE 建模正常视频                              | 时空重构分支代表                     | 编码-解码视频片段并用重构误差检测异常                        | 结构简洁                 | 对复杂背景鲁棒性有限       |
| 无监督/时空 AE              | Spatio-Temporal AutoEncoder for Video Anomaly Detection      |     2017 | ACM MM workshop/相关 | 联合空间与时间重构正常模式                            | 说明 AE 从图像重构扩展到视频时空重构 | 以 3D/时序结构捕获正常片段                                   | 适合小规模正常训练       | 泛化到复杂异常较弱         |
| 无监督/预测                 | Future Frame Prediction for Anomaly Detection: A New Baseline |     2018 | CVPR，CCF A          | 将未来帧预测明确建立为 VAD 基线                       | prediction-based VAD 必引            | 预测下一帧并以预测误差判别异常                               | 对动态异常敏感           | 易受遮挡和背景运动影响     |
| 无监督/外观运动一致性       | Anomaly Detection in Video Sequence with Appearance-Motion Correspondence |     2019 | ICCV，CCF A          | 利用外观与运动对应关系发现异常                        | 重构/一致性混合路线                  | 比较 RGB 外观与运动流之间的一致性                            | 降低单模态误报           | 依赖光流质量               |
| 无监督/骨架轨迹             | Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos |     2019 | CVPR，CCF A          | 用 skeleton trajectories 学习人体运动规律             | 骨架轨迹预测代表                     | 以人体关键点轨迹建模正常运动                                 | 对人体行为异常敏感       | 依赖姿态估计质量           |
| 无监督/记忆库               | Memorizing Normality to Detect Anomaly: Memory-Augmented Deep Autoencoder |     2019 | ICCV，CCF A          | 用 memory items 存储正常原型                          | 记忆增强 normality modeling 奠基     | AE 查询正常记忆项，异常难以匹配                              | 缓解 AE 过强重构         | 记忆覆盖不足会误报         |
| 无监督/对象级重构           | Object-centric Auto-encoders and Dummy Anomalies for Abnormal Event Detection in Video |     2019 | CVPR，CCF A          | 将重构从整帧转向对象级目标                            | 对象级 VAD 代表                      | 检测目标并对对象 patch/轨迹建模                              | 减少背景干扰             | 依赖检测器质量             |
| 无监督/记忆原型             | Learning Memory-Guided Normality for Anomaly Detection       |     2020 | CVPR，CCF A          | 引入 memory module 与 compactness/separateness loss   | 正常原型与特征紧致性核心代表         | 用记忆原型约束正常特征分布                                   | 可建模多样正常模式       | 场景迁移仍有限             |
| 无监督/姿态图聚类           | Graph Embedded Pose Clustering for Anomaly Detection         |     2020 | CVPR，CCF A          | 用姿态图嵌入和聚类检测异常动作                        | pose graph/prototype 分支            | 将人体姿态映射到图嵌入空间并聚类                             | 对人体异常解释性较强     | 非人体异常覆盖不足         |
| 无监督/聚类 AE              | Clustering Driven Deep Autoencoder for Video Anomaly Detection |     2020 | ECCV，CCF B          | 聚类驱动 AE 学习多模态正常模式                        | 正常多样性建模                       | 结合聚类约束与 AE 重构                                       | 能处理多正常模式         | 聚类质量影响性能           |
| 无监督/预测                 | Future Frame Prediction Network for Video Anomaly Detection  |     2021 | TPAMI/相关期刊       | 改进未来帧预测网络                                    | 预测式方法延伸                       | 优化未来帧预测以增强异常残差                                 | 延续经典预测范式         | 仍依赖像素级预测质量       |
| 无监督/生成式               | Generative Cooperative Learning for Unsupervised Video Anomaly Detection |     2022 | CVPR，CCF A          | 生成式协同学习建模正常分布                            | 生成式无监督 VAD 代表                | 多模型协同学习正常生成和判别                                 | 生成分布表达更强         | 训练复杂                   |
| 自监督/Masked AE            | Detecting Anomalous Events from Unlabeled Videos via Temporal Masked Auto-Encoding |     2022 | 会议/期刊            | 用 temporal masked auto-encoding 学习正常时序         | 自监督 VAD 代表                      | 掩码视频片段并学习恢复正常动态                               | 减少人工标签依赖         | 对异常语义解释有限         |
| 无监督/扩散模型             | Feature Prediction Diffusion Model for Video Anomaly Detection |     2023 | 会议/期刊            | 将扩散模型用于特征预测式 VAD                          | 扩散式 VAD 近年代表                  | 通过扩散过程建模正常特征分布                                 | 生成建模能力强           | 推理成本较高               |
| 弱监督/MIL 奠基             | Real-World Anomaly Detection in Surveillance Videos          |     2018 | CVPR，CCF A          | 提出 UCF-Crime 与 MIL ranking 弱监督范式              | 弱监督 VAD 奠基                      | 用视频级标签训练片段异常排序模型                             | 大规模真实监控           | 弱标签噪声大               |
| 弱监督/MIL 改进             | CLAWS: Clustering Assisted Weakly Supervised Learning with Normalcy Suppression |     2020 | ECCV，CCF B          | normalcy suppression 与聚类辅助弱监督                 | MIL 后续改进                         | 聚类增强异常片段选择并抑制正常性                             | 提高异常/正常分离        | 依赖聚类与伪标签           |
| 弱监督/多模态               | Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision |     2020 | ECCV，CCF B          | 提出 XD-Violence，融合音频与视觉弱监督                | 多模态弱监督基准                     | 音视频特征联合学习暴力/异常片段                              | 利用音频互补信息         | 模态噪声和不同步明显       |
| 弱监督/判别学习             | Weakly Supervised Video Anomaly Detection via Center-Guided Discriminative Learning |     2020 | ICME/相关            | 中心引导增强异常/正常判别性                           | 特征判别弱监督                       | 学习类别中心并扩大特征间隔                                   | 判别边界更清晰           | 仍依赖视频级标签           |
| 弱监督/自训练               | MIST: Multiple Instance Self-Training Framework for Video Anomaly Detection |     2021 | CVPR，CCF A          | 多实例自训练优化伪标签                                | 伪标签/self-training 代表            | 迭代挖掘异常片段并更新模型                                   | 改善弱标签定位           | 初始伪标签偏差会传播       |
| 弱监督/特征幅值             | Weakly-Supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning |     2021 | ICCV，CCF A          | 用 temporal feature magnitude 提升弱监督鲁棒性        | 片段判别与鲁棒性                     | 假设异常片段特征幅值更显著                                   | 简洁有效                 | 对特征尺度敏感             |
| 弱监督/协同正常性           | Collaborative Normality Learning Framework for Weakly Supervised Video Anomaly Detection |     2022 | 会议/期刊            | 协同学习正常模式辅助弱监督定位                        | 弱监督正常性约束                     | 正常学习与异常判别联合优化                                   | 缓解异常标签稀疏         | 模型结构较复杂             |
| 弱监督/音视频对比           | Modality-Aware Contrastive Instance Learning with Self-Distillation for Weakly-Supervised Audio-Visual Violence Detection |     2022 | ACM MM，CCF A        | 音视频对比实例学习与自蒸馏                            | 多模态弱监督代表                     | 学习模态感知实例表示并蒸馏                                   | 提升模态互补利用         | 对缺失模态敏感             |
| 弱监督/Transformer          | Self-Training Multi-Sequence Learning with Transformer for Weakly Supervised Video Anomaly Detection |     2022 | 会议/期刊            | Transformer 与多序列自训练                            | 时序建模弱监督补充                   | 通过 Transformer 捕获长程片段关系                            | 长程依赖更强             | 训练成本增加               |
| 弱监督/音视频融合           | Weakly Supervised Audio-Visual Violence Detection            |     2022 | TMM/相关期刊         | 音视频融合暴力异常检测                                | 多模态弱监督补充                     | 融合视觉与音频判别暴力事件                                   | 对听觉异常敏感           | 泛化到非暴力异常有限       |
| 弱监督/时序关系             | Weakly Supervised Video Anomaly Detection via Transformer-Enabled Temporal Relation Learning |     2022 | 会议/期刊            | 用 Transformer 建模片段时间关系                       | 时序关系建模                         | 构建片段间时序依赖并输出异常分数                             | 长视频上下文更充分       | 不提供因果解释             |
| 弱监督/标签噪声             | Weakly-Supervised Anomaly Detection in Video Surveillance via Graph Convolutional Label Noise Cleaning |     2022 | 期刊/会议            | 图卷积清洗弱标签噪声                                  | 弱监督噪声处理                       | 建图传播标签可信度并清洗噪声                                 | 缓解粗标签污染           | 图结构构建影响结果         |
| 弱监督/伪标签               | Exploiting Completeness and Uncertainty of Pseudo Labels for Weakly Supervised Video Anomaly Detection |     2023 | ICCV/相关            | 用完整性与不确定性优化伪标签                          | 伪标签优化代表                       | 估计伪标签完整性和不确定性                                   | 边界定位更稳             | 估计误差会影响训练         |
| 弱监督/正常性引导           | Normality Guided Multiple Instance Learning for Weakly Supervised Video Anomaly Detection |     2023 | 会议/期刊            | 正常性引导 MIL 学习                                   | MIL 正常约束代表                     | 将正常片段建模纳入 MIL                                       | 减少异常误检             | 正常多样性仍难覆盖         |
| 弱监督/无偏 MIL             | Unbiased Multiple Instance Learning for Weakly Supervised Video Anomaly Detection |     2023 | CVPR/相关            | 缓解 MIL 片段选择偏置                                 | 弱监督偏置问题                       | 修正 MIL 对高分片段的偏置学习                                | 更公平定位异常           | 依赖偏置建模假设           |
| 弱监督/Prompt               | Prompt-Enhanced Multiple Instance Learning for Weakly Supervised Video Anomaly Detection |     2024 | CVPR，CCF A          | abnormal-aware 与 normal context prompts 注入语义先验 | 连接传统弱监督与 VLM prompt          | 将文本提示与 MIL 片段学习结合                                | 语义先验增强             | Prompt 设计敏感            |
| 弱监督/时序判别 Transformer | Weakly Supervised Video Anomaly Detection via Self-Guided Temporal Discriminative Transformer |     2024 | 会议/期刊            | 自引导时序判别 Transformer                            | 近年时序优化                         | 学习时序判别特征并自引导定位                                 | 时序判别性强             | 推理/训练成本较高          |
| 弱监督/文本正常性引导       | Text Prompt with Normality Guidance for Weakly Supervised Video Anomaly Detection |     2024 | 会议/期刊            | 文本提示与正常性引导弱监督 VAD                        | VLM 语义先验补充                     | 用文本提示描述正常/异常语义                                  | 开放语义潜力             | 文本提示可靠性待验证       |
| 全监督/空间定位评价         | Anomaly Detection and Localization in Crowded Scenes         |     2014 | TPAMI，CCF A         | 系统化 frame/pixel-level 定位评价                     | 全监督/空间定位评价依据              | 使用像素级标注评价异常区域定位                               | 指标影响深远             | 评价协议可能过宽松         |
| 全监督/行为基准             | Abnormal Event Detection at 150 FPS in MATLAB                |     2013 | ICCV，CCF A          | Avenue 提供行为异常标注和检测基准                     | 小规模监督评价基准                   | 用标注异常片段测试检测方法                                   | 简洁经典                 | 类别与场景有限             |
| 全监督/多场景基准           | A Revisit of Sparse Coding Based Anomaly Detection in Stacked RNN Framework |     2017 | ICCV，CCF A          | ShanghaiTech 推动多场景 VAD 评测                      | 多场景监督/弱监督评测基础            | 多摄像头校园异常数据                                         | 场景更丰富               | 仍以检测定位为主           |
| 开放世界/驾驶异常           | When, Where, and What? A New Dataset for Anomaly Detection in Driving Videos |     2020 | 会议/数据集论文      | DoTA 提供驾驶异常时间、空间、类别标注                 | 交通开放场景 VAD                     | 面向驾驶视频进行时空类别异常检测                             | 支持 when/where/what     | 领域限定为驾驶             |
| 开放世界/评价协议           | Street Scene: A New Dataset and Evaluation Protocol for Video Anomaly Detection |     2020 | WACV/相关            | 提出 Street Scene 与 RBDC/TBDC                        | 评价协议反思                         | 更关注区域和轨迹级检测质量                                   | 更贴近实际部署           | WACV 不属主流 CCF A/B      |
| 开放世界/多模态基准         | Not only Look, but also Listen: Learning Multimodal Violence Detection under Weak Supervision |     2020 | ECCV，CCF B          | XD-Violence 提供大规模音视频弱监督异常基准            | 开放场景多模态异常基准               | 真实多场景暴力/异常视频                                      | 规模大、模态多           | 弱标签粒度粗               |
| 开放集监督                  | Towards Open Set Video Anomaly Detection                     |     2022 | ECCV workshop/相关   | 探索开放集 VAD 设定                                   | 开放集问题定义                       | 训练和测试异常类别不完全一致                                 | 直面未知异常             | 体系仍早期                 |
| 全监督/开放集基准           | UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection |     2022 | CVPR，CCF A          | 提出 supervised open-set VAD benchmark                | 全监督开放集核心引用                 | 合成场景中分离训练/测试异常类型                              | 支持开放集评测           | 合成域与真实域存在差距     |
| 开放词汇 VAD                | Open-Vocabulary Video Anomaly Detection                      |     2024 | CVPR，CCF A          | 类别无关检测与类别相关识别                            | 开放世界 VAD 核心代表                | 使用大模型语义知识识别 unseen anomalies                      | 开放类别能力强           | 依赖文本/生成先验          |
| VAU 开放理解基准            | Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly |     2024 | CVPR，CCF A          | CUVA 推动异常从检测走向因果理解                       | 开放世界理解边界                     | 标注异常类型、描述、原因、后果                               | 支持 VAU 评价            | 理解指标仍早期             |
| 大模型/VFM                  | VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking |     2023 | CVPR，CCF A          | 扩展视频 masked autoencoder 预训练                    | VFM 视觉表征底座                     | 通过双重掩码进行大规模视频自监督预训练                       | 视频表征泛化强           | 非专门 VAD 方法            |
| 大模型/VFM                  | InternVideo: General Video Foundation Models via Generative and Discriminative Learning |     2023 | CVPR，CCF A          | 生成式与判别式学习构建通用视频基础模型                | 视频基础模型背景                     | 大规模视频预训练得到通用特征                                 | 迁移能力强               | VAD 需任务适配             |
| 大模型/VFM/多模态表征       | ImageBind: One Embedding Space To Bind Them All              |     2023 | CVPR，CCF A          | 构建多模态统一嵌入空间                                | 多模态异常理解底座                   | 将图像、文本、音频等模态对齐                                 | 支持跨模态融合           | 非异常专用                 |
| 大模型/VFM/分割基础模型     | Segment Anything                                             |     2023 | ICCV，CCF A          | 提出通用视觉分割基础模型 SAM                          | 空间定位与对象级异常分析底座         | Promptable segmentation 支持对象区域提取                     | 空间泛化强               | 不具备视频因果理解         |
| 大模型/VLM 弱监督           | VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection |     2024 | AAAI，CCF A          | 将 CLIP 适配到弱监督 VAD                              | VLM 进入 VAD 核心引用                | 结合视觉分支与视觉语言对齐分支                               | 语义先验强               | Prompt 与域适配敏感        |
| 大模型/开放词汇             | Open-Vocabulary Video Anomaly Detection                      |     2024 | CVPR，CCF A          | 同时处理类别无关检测与类别相关识别                    | 闭集 VAD 到开放词汇                  | 利用大模型语义知识与合成 unseen anomalies                    | 开放泛化强               | 类别描述质量影响识别       |
| 大模型/VLM+LLM              | Harnessing Large Language Models for Training-free Video Anomaly Detection |     2024 | CVPR，CCF A          | LAVAD：VLM caption + LLM 训练自由异常评分             | training-free VAD 代表               | 生成帧级文本描述并由 LLM 聚合异常分数                        | 无需训练、语义可读       | 受 caption 与 LLM 幻觉影响 |
| 大模型/Prompt-MIL           | Prompt-Enhanced Multiple Instance Learning for Weakly Supervised Video Anomaly Detection |     2024 | CVPR，CCF A          | 用异常感知 prompt 增强 MIL                            | VLM prompt 与弱监督结合              | 将正常上下文 prompt 与异常 prompt 注入片段学习               | 语义辅助定位             | 仍以 VAD 分数为主          |
| 大模型/VFM-MAE              | Self-Distilled Masked Auto-Encoders are Efficient Video Anomaly Detectors |     2024 | CVPR，CCF A          | 自蒸馏 MAE 高效用于 VAD                               | MAE/VFM 式 VAD 核心代表              | motion-gradient token weighting 与 decoder discrepancy       | 高效、表征强             | 仍偏检测定位               |
| 大模型/VFM 自监督           | Multi-Scale Video Anomaly Detection by Multi-Grained Spatio-Temporal Representation Learning |     2024 | CVPR，CCF A          | 多粒度时空 proxy task 学习 VAD 表征                   | 自监督时空表征代表                   | continuity judgment、discontinuity localization、missing frame estimation | 捕获小尺度异常           | 语义解释有限               |
| 大模型/VAU 因果理解         | Uncovering What, Why and How: A Comprehensive Benchmark for Causation Understanding of Video Anomaly |     2024 | CVPR，CCF A          | 提出 CUVA 与 MMEval                                   | VAU benchmark 核心                   | 标注异常 what/why/how、原因和后果                            | 直接支撑 VAU 定义        | 评价仍需成熟               |
| 大模型/长视频理解           | MovieChat: From Dense Token to Sparse Memory for Long Video Understanding |     2024 | CVPR，CCF A          | 用稀疏记忆支持长视频理解                              | 长视频 VAU 技术参考                  | 将长视频压缩为可查询记忆                                     | 适合长视频上下文         | 非异常专用                 |
| 大模型/时间敏感 MLLM        | TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding |     2024 | CVPR，CCF A          | 面向长视频时间理解的 MLLM                             | temporal grounding 参考              | 结合时间戳和视频问答进行长视频理解                           | 时间意识更强             | 异常因果需适配             |
| 大模型/视频时刻定位         | VTimeLLM: Empower LLM to Grasp Video Moments                 |     2024 | CVPR，CCF A          | 让 LLM 进行视频片段时间定位                           | VAU grounding 技术参考               | 将视频时刻定位转化为语言模型可处理任务                       | 有助于异常片段 grounding | 非 VAD 专用                |
| 大模型/长视频多粒度 VAU     | Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity |     2025 | CVPR，CCF A          | 提出长视频、多粒度 VAU 数据与模型                     | VAU 近年核心方向                     | 覆盖 clip/event/video 级异常理解并使用异常聚焦采样           | 更贴近长视频理解         | 计算成本和可靠性待验证     |

## 4. 专题：传统深度学习与大模型 VAD 趋势总结

### 4.1 传统深度学习 VAD 整体发展趋势

传统深度学习 VAD 的第一条趋势是从“整帧正常性建模”走向“对象、轨迹和结构化正常性建模”。早期方法多使用整帧重构误差或全局特征距离，后续逐步转向对象级 AE、骨架轨迹、pose graph clustering 和 appearance-motion consistency，以降低背景干扰并增强行为层判别。

第二条趋势是从“单一重构/预测误差”走向“多机制融合”。重构式方法容易重构异常，预测式方法容易受背景运动影响，度量/记忆方法依赖正常覆盖，生成式方法训练成本较高。因此近年工作常将重构、预测、记忆、聚类、图结构和自监督 proxy task 结合起来，提高复杂场景下的鲁棒性。

第三条趋势是从“无监督正常训练”走向“弱监督大规模真实视频”。UCF-Crime 与 XD-Violence 推动 VAD 进入真实监控长视频和视频级标签阶段，MIL、伪标签、自训练、Transformer 和多模态融合成为弱监督主线。该路线降低标注成本，但也把噪声传播、边界模糊和类别偏置变成核心难题。

第四条趋势是从“封闭场景检测”走向“开放集与评价协议反思”。UBnormal、Street Scene、DoTA 等工作说明，仅用 frame-level AUC 衡量 VAD 已不足以反映真实部署需求。未来传统 VAD 仍需要更强的开放集泛化、空间定位评价、低延迟部署和隐私保护能力。

### 4.2 大模型 VAD/VAU 近年发展趋势

大模型 VAD/VAU 的第一条趋势是 VFM 提升视觉表征迁移。VideoMAE V2、InternVideo、SAM、ImageBind 以及 MAE 式 VAD 工作说明，大规模预训练视觉表征可以改善小样本、跨场景和多尺度异常检测，但 VFM 本身主要解决“看得更稳”，不直接解决“解释得更对”。

第二条趋势是 VLM 推动开放词汇异常识别。VadCLIP、Prompt-Enhanced MIL、Open-Vocabulary VAD 和 LAVAD 将文本提示、视觉语言对齐和 LLM 推理引入异常检测，使模型开始具备类别语义匹配、训练自由评分或 unseen anomaly recognition 能力。该路线把 VAD 从封闭二分类推向开放词汇识别，但 prompt 敏感性、类别描述质量和语言先验偏置仍是风险。

第三条趋势是 MLLM 推动 VAU 形成独立任务。CUVA 与 Holmes-VAU 将研究重点从“异常检测定位”扩展到 what/why/how、事件描述、原因解释、后果推理和多粒度长视频理解。MovieChat、TimeChat、VTimeLLM 等长视频 MLLM 为 VAU 的 temporal grounding 提供方法参考，但异常场景下的因果忠实性、幻觉控制和评价标准仍未成熟。

第四条趋势是大模型暴露出新的系统瓶颈。相比传统 VAD，大模型带来开放语义和交互解释能力，但同时引入高计算成本、长视频上下文压缩、隐私安全、部署延迟、模型幻觉和不可验证解释等问题。因此，未来 VAD/VAU 的关键不只是“接入更大的模型”，而是构建可信、可评估、可部署的异常理解系统。

### 4.3 传统方法 vs VFM vs VLM vs MLLM 多维对比

| 维度 | 传统深度学习 VAD | VFM 路线 | VLM 路线 | MLLM/VAU 路线 |
|---|---|---|---|---|
| 检测能力 | 在固定场景、封闭类别和标准数据集上成熟；弱监督可处理长视频 | 视觉表征更强，跨场景迁移优于小模型 | 可结合文本先验提升异常类别识别 | 可在检测后进行问答和解释，但定位稳定性仍弱 |
| 语义理解 | 通常只输出分数、边界或热力图 | 主要提升视觉特征，不直接生成语义解释 | 可进行文本类别匹配和开放词汇识别 | 可生成描述、原因、后果和严重程度判断 |
| 开放泛化 | 较弱，依赖训练场景和正常模式覆盖 | 中等，预训练表征改善迁移 | 较强，文本提示支持 unseen categories | 理论最强，可处理开放问答，但可靠性不稳定 |
| 标注成本 | 无监督低，弱监督中等，全监督高 | 预训练成本高，任务标注可减少 | 可零样本/少样本，但 prompt 设计需要人工知识 | 可少样本交互，但高质量 VAU 数据仍稀缺 |
| 推理速度 | 通常最快，适合边缘部署 | 中等，取决于 backbone 大小 | 较慢，需视觉-语言编码和文本匹配 | 最慢，长视频和语言生成成本高 |
| 可解释性 | 多为误差、热力图、轨迹属性，解释有限 | 可通过特征/分割辅助定位 | 可输出类别语义和文本匹配依据 | 可生成自然语言解释和因果链，但有幻觉风险 |
| 部署难度 | 较低到中等，工程成熟 | 中等，需要大 backbone 或蒸馏压缩 | 中高，需要跨模态模型和 prompt 管理 | 高，需要长视频处理、LLM 服务、安全与隐私控制 |
| 主要瓶颈 | normality 假设、弱标签噪声、跨场景泛化 | 语义和因果能力不足 | prompt 敏感、语言先验偏置、定位不稳 | 幻觉、因果忠实性、实时性、评价标准不成熟 |
| 适合任务 | 检测、评分、时间/空间定位 | 表征迁移、少样本 VAD、多尺度检测 | 开放词汇异常识别、语义辅助弱监督 | 异常描述、问答、因果解释、后果推理 |

