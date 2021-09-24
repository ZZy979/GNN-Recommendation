# 基于知识图谱的推荐算法
## 数据集
oag-cs - 使用OAG微软学术数据构造的计算机领域的学术网络（见 [readme](data/readme.md)）

## 预处理
### 预训练顶点嵌入
使用metapath2vec（随机游走+word2vec）预训练顶点嵌入，作为GNN模型的顶点输入特征
1. 随机游走 `python -m gnnrec.kgrec.preprocess.random_walk model/word2vec/oag_cs_corpus.txt`
2. 训练词向量 `python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 model/word2vec/oag_cs_corpus.txt model/word2vec/oag_cs.model`

### 获取顶点表示向量
TODO 训练GNN模型，获取顶点的表示向量，GNN模型使用异构图表示学习模块改进的RHCO模型

## 召回
使用微调后的SciBERT模型（见 [readme](data/readme.md) 第2步）将查询词编码为向量，与预先计算好的论文标题向量计算余弦相似度，取top k

`python -m gnnrec.kgrec.recall data/oag/cs/paper_feat.pkl model/scibert.pt data/oag/cs/mag_papers.txt`

召回结果示例：

graph neural network
```
0.959662914276123 On the choice of graph neural network architectures
0.9582182168960571 Neural Graph Machines: Learning Neural Networks Using Graphs
0.9409174919128418 Neural Graph Learning: Training Neural Networks Using Graphs
0.9358110427856445 Measuring and Improving the Use of Graph Information in Graph Neural Networks
0.9337150454521179 Supervised Neural Network Models for Processing Graphs
0.9323418140411377 Computational Capabilities of Graph Neural Networks
0.9271646738052368 Challenging the generalization capabilities of Graph Neural Networks for network modeling
0.9247722625732422 Geometrically Principled Connections in Graph Neural Networks
0.9156403541564941 Hyper-SAGNN: a self-attention based graph neural network for hypergraphs
0.9113651514053345 What graph neural networks cannot learn: depth vs width
```

recommendation algorithm based on knowledge graph
```
0.91325843334198 Research on Video Recommendation Algorithm Based on Knowledge Reasoning of Knowledge Graph
0.8992738723754883 An Improved Recommendation Algorithm in Knowledge Network
0.8892782330513 A personalized recommendation algorithm based on interest graph
0.876215398311615 An Improved Recommendation Algorithm Based on Graph Model
0.8555296659469604 Recommendation Algorithm Based on Graph-Model Considering User Background Information
0.8538336753845215 The Research of Recommendation Algorithm based on Complete Tripartite Graph Model
0.8433443307876587 Studying Recommendation Algorithms by Graph Analysis
0.8413041830062866 WMR--A Graph-Based Algorithm for Friend Recommendation
0.8361994028091431 Evaluating Recommendation Algorithms by Graph Analysis
0.8265330791473389 A Tripartite Graph Recommendation Algorithm Based on Item Information and User Preference
```

Drone swarm
```
0.9370608925819397 Design and simulation of the emergent behavior of small drones swarming for distributed target localization
0.9356381893157959 An innovative tri-rotor drone and associated distributed aerial drone swarm control
0.9218730926513672 Cooperative comodule discovery for swarm-intelligent drone arrays
0.9094085693359375 Intelligent Drone Swarm for Search and Rescue Operations at Sea.
0.9033511877059937 A new approach to realize drone swarm using ad-hoc network
0.8862361907958984 Control of a Remote Swarm of Drones/Robots Through a Local (Possibly Model) Swarm: Qualitative and Quantitative Issues
0.8621711730957031 Investigating the survivability of drone swarms with flocking and swarming flight patterns using Virtual Reality
0.8603955507278442 Using Optimization, Learning, and Drone Reflexes to Maximize Safety of Swarms of Drones
0.8571282625198364 Autonomous UAV Swarm: Behavior Generation and Simulation
0.855229377746582 Swarm of Networked Drones for Video Detection of Intrusions
```

没有微调的SciBERT召回结果：

grapu neural network
```
0.23215869069099426 Video-Quality Estimation Based on Reduced-Reference Model Employing Activity-Difference
0.21934255957603455 Methods to summarize change among land categories across time intervals
0.21367816627025604 Optical signal processing in transverse acousto-optic waveguide-type functional devices with SAW
0.21230876445770264 In-pixel time-to-digital converter for 3D TOF cameras with time amplifier
0.21213743090629578 Excitation modeling based on speech residual information
0.21198226511478424 Recovery of signals with time-varying spectral support based on the modulated wideband converter
0.2099592536687851 Performance Improvements in Sub-Band Coding Using the Proposed ADM
0.20989662408828735 Wavelength Conversion with 2R-Regeneration by UL-SOA Induced Chirp Filtering
0.2095218002796173 Methods for linking EHR notes to education materials
0.20687958598136902 Improved DOA estimator for wideband sources using two references
```

recommendation algorithm based on knowledge graph
```
0.2637781798839569 Limits to the mechanization of the detailing step paradigm
0.25841644406318665 Effect of simultaneous vibrations to two tendons on velocity of the induced illusory movement
0.25569063425064087 Translation using Information on Dialogue Participants
0.25277456641197205 Effect of the nerve fiber path eccentricity on the single fiber action potential
0.2520480155944824 Fixed-Cost Pooling Strategies Based on IR Evaluation Measures
0.2498779594898224 Proposal of deleting plots from the reviews to the items with stories
0.24886372685432434 Temporal residual data sub-sampling in LDV representation format
0.24797609448432922 Trial sequential boundaries for cumulative meta-analyses
0.24725483357906342 Readin', Writtin' [sic], 'Rithmetic: Reference Desk Redux
0.24669909477233887 Techniques for scaling up analyses based on pre-interpretations
```
