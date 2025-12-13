# Attention Is All You Need

> 论文：Vaswani et al., “Attention Is All You Need”
> 核心：提出Transformer——完全用注意力机制替代RNN/CNN的序列到序列架构，在机器翻译上更准、更快、更易并行

---

## 1. 论文核心概念（核心insight总结）

Transformer 的核心 insight 可以概括为三句话：

1. **序列建模不一定需要“按时间步递归”**。RNN 的计算天然是串行的，长序列训练难并行；Transformer 用自注意力一次性把所有位置两两建立依赖，从而把训练中的“时间维串行”几乎消掉。

2. **用自注意力（Self-Attention）直接建模全局依赖**。任意两个位置的关系在一层自注意力里就能连通，最大路径长度是常数级（对比RNN是O(n)）。

3. **多头注意力（Multi-Head）抵消“注意力加权平均会丢分辨率”的副作用**：把不同“子空间/模式”的对齐并行学出来再拼接，提升表达力与可解释性。

---

## 2. 论文内容词解释（重要名词详细解释）

### 2.1 Sequence Transduction / Seq2Seq（序列转导/序列到序列）

输入序列 ((x_1,\dots,x_n)) 映射到输出序列 ((y_1,\dots,y_m)) 的任务范式，如机器翻译、摘要生成。典型结构是 **Encoder-Decoder**：编码器把输入编码成表示 (z)，解码器自回归地产生输出。

### 2.2 Encoder / Decoder（编码器/解码器）

* **Encoder**：把输入token嵌入后，经过多层变换，输出一串上下文表示 (z=(z_1,\dots,z_n))
* **Decoder**：在生成第 (i) 个输出时，只能依赖已生成的 (<i) 的输出（自回归），并可通过“encoder-decoder attention”访问输入表示 (z)

### 2.3 Attention（注意力）

把 **Query** 和一组 **Key-Value** 映射成输出：输出是对 Value 的加权和，权重由 Query 和 Key 的“相似度/兼容度”决定。

### 2.4 Self-Attention（自注意力 / Intra-Attention）

同一序列内部各位置互相“看”，即 Q/K/V 都来自同一个序列表示（上一层输出）。它让每个位置能直接聚合整个序列的信息。

### 2.5 Scaled Dot-Product Attention（缩放点积注意力）

论文定义的注意力核心算子（也是今天Transformer家族的基础）：
[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
]
其中缩放 (\frac{1}{\sqrt{d_k}}) 用来避免 (d_k) 大时点积过大导致 softmax 梯度很小的问题。

### 2.6 Multi-Head Attention（多头注意力）

把同一输入通过不同线性投影得到 (h) 组 Q/K/V，在每个“头”上并行做注意力，再拼接并投影：
[
\mathrm{head}_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
]
[
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O
]

### 2.7 Positional Encoding（位置编码）

因为 Transformer **既没有RNN的时间递归，也没有CNN的卷积邻域**，需要显式注入位置信息。论文用固定的正弦/余弦位置编码：
[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
]
特点：可外推到更长序列；相对位移 (k) 具有良好的线性可表示性（论文给出直觉动机）。

### 2.8 Residual Connection + LayerNorm（残差连接+层归一化）

每个子层输出是：
[
\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))
]
残差保证深层训练稳定，LayerNorm 提升优化与收敛。

### 2.9 Masked Self-Attention（带掩码的解码器自注意力）

解码器的自注意力必须防止“偷看未来”，通过把非法连接在 softmax 前置为 (-\infty) 实现因果掩码。

### 2.10 BLEU / PPL（机器翻译指标/困惑度）

* **BLEU**：机器翻译常用自动评估指标（n-gram匹配）。论文用它报告 WMT14 EN-DE/EN-FR 成绩。
* **PPL（perplexity）**：语言模型/生成模型常用指标；论文在架构变体实验里报告 dev perplexity 与 BLEU 的变化。

---

## 3. 论文方法

### 3.1 过去方法的问题（motivation）

论文主要对比两类主流序列模型：

**(1) RNN/LSTM/GRU 的问题：串行计算**
RNN 按位置 (t) 递推隐藏状态 (h_t=f(h_{t-1},x_t))，训练时每个样本内部无法并行（只能跨样本batch并行），序列越长越难提速。

**(2) CNN-based Seq2Seq 的问题：远距离依赖路径长**
卷积能并行，但若核宽 (k<n)，要让两个远位置交互，需要堆很多层（路径长度随距离增长），学习长距离依赖更难。

**Transformer 的动机**

* 用自注意力让任意两位置在一层内直接交互（路径短）
* 同时整个层内是矩阵乘法 + softmax，适合GPU大规模并行

---

### 3.2 整体框架（可复现级别的流程说明）

下面按“你要复现一个标准Transformer（论文base配置）”的角度，把关键模块、张量形状、公式、超参都串起来。论文的整体结构是典型 Encoder-Decoder，但每层由 **(自注意力 + FFN)** 组成。

#### 3.2.1 符号与维度约定

* 输入序列长度：(n)，输出序列长度：(m)
* 模型维度：(d_{model}=512)（base）
* 多头数：(h=8)（base）
* 每头维度：(d_k=d_v=d_{model}/h=64)
* FFN 隐藏维：(d_{ff}=2048)（base）

把输入 token 先映射到 embedding：

* Encoder输入嵌入：(X\in\mathbb{R}^{n\times d_{model}})
* Decoder输入嵌入（右移一位的目标序列）：(Y\in\mathbb{R}^{m\times d_{model}})

再加上位置编码：
[
X \leftarrow X + PE,\quad Y \leftarrow Y + PE
]
位置编码采用正弦/余弦方案。

#### 3.2.2 Scaled Dot-Product Attention 的矩阵实现（核心算子）

给定：

* (Q\in\mathbb{R}^{L_q\times d_k})
* (K\in\mathbb{R}^{L_k\times d_k})
* (V\in\mathbb{R}^{L_k\times d_v})

计算：
[
A=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)\in\mathbb{R}^{L_q\times L_k}
]
[
\mathrm{Attention}(Q,K,V)=AV\in\mathbb{R}^{L_q\times d_v}
]
对应论文公式(1)。

> **解码器掩码**：对未来位置的 logits 加 (-\infty)，使 softmax 后权重为0，实现“只能看左边”。

#### 3.2.3 Multi-Head Attention（可复现的参数与步骤）

对第 (i) 个头，有可学习矩阵：

* (W_i^Q\in\mathbb{R}^{d_{model}\times d_k})
* (W_i^K\in\mathbb{R}^{d_{model}\times d_k})
* (W_i^V\in\mathbb{R}^{d_{model}\times d_v})

以及输出投影：

* (W^O\in\mathbb{R}^{(h\cdot d_v)\times d_{model}})

计算流程（以输入 (H\in\mathbb{R}^{L\times d_{model}}) 为例）：

1. 线性投影：(Q_i=HW_i^Q,\ K_i=HW_i^K,\ V_i=HW_i^V)
2. 每头注意力：(\mathrm{head}_i=\mathrm{Attention}(Q_i,K_i,V_i)\in\mathbb{R}^{L\times d_v})
3. 拼接：(\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\in\mathbb{R}^{L\times (h\cdot d_v)})
4. 输出：(\mathrm{MHA}(H)=\mathrm{Concat}(\cdots)W^O\in\mathbb{R}^{L\times d_{model}})

#### 3.2.4 Position-wise Feed-Forward Network（逐位置前馈网络）

每层都有一个对每个位置独立同构的两层MLP：
[
\mathrm{FFN}(x)=\max(0, xW_1+b_1)W_2+b_2
]
其中：

* (W_1\in\mathbb{R}^{d_{model}\times d_{ff}}),
* (W_2\in\mathbb{R}^{d_{ff}\times d_{model}})

#### 3.2.5 Encoder Layer（单层结构与复现步骤）

论文 base encoder 堆叠 (N=6) 层。单层包含两个子层：

给定输入 (H^{(\ell-1)}\in\mathbb{R}^{n\times d_{model}})：

1. 自注意力子层
   [
   \tilde{H} = \mathrm{LayerNorm}\left(H^{(\ell-1)}+\mathrm{Dropout}(\mathrm{MHA}(H^{(\ell-1)}))\right)
   ]
2. FFN 子层
   [
   H^{(\ell)}=\mathrm{LayerNorm}\left(\tilde{H}+\mathrm{Dropout}(\mathrm{FFN}(\tilde{H}))\right)
   ]

残差+LayerNorm 的形式在论文中给出，dropout 位置在训练章节描述（对子层输出、embedding+PE 都做）。

#### 3.2.6 Decoder Layer（单层结构与复现步骤）

decoder 也堆叠 (N=6) 层，但每层有三个子层：

给定 decoder 输入 (S^{(\ell-1)}\in\mathbb{R}^{m\times d_{model}}) 与 encoder 输出 (Z\in\mathbb{R}^{n\times d_{model}})：

1. **Masked Self-Attention（因果掩码）**
   [
   \hat{S}=\mathrm{LayerNorm}\left(S^{(\ell-1)}+\mathrm{Dropout}(\mathrm{MHA_{mask}}(S^{(\ell-1)}))\right)
   ]
   论文解释了为何需要mask以保证自回归。

2. **Encoder-Decoder Attention（交叉注意力）**
   Query 来自 decoder，Key/Value 来自 encoder 输出 (Z)：
   [
   \tilde{S}=\mathrm{LayerNorm}\left(\hat{S}+\mathrm{Dropout}(\mathrm{MHA}(\hat{S}, Z, Z))\right)
   ]

3. **FFN**
   [
   S^{(\ell)}=\mathrm{LayerNorm}\left(\tilde{S}+\mathrm{Dropout}(\mathrm{FFN}(\tilde{S}))\right)
   ]

#### 3.2.7 输出层（Softmax & 权重共享）

decoder 顶层输出 (S^{(N)}\in\mathbb{R}^{m\times d_{model}}) 通过线性层 + softmax 得到每个位置的下一token分布。论文还做了 **embedding 与 pre-softmax 权重共享**，并在 embedding 处乘 (\sqrt{d_{model}})。

#### 3.2.8 训练细节（复现实验的关键超参）

* 数据：WMT14 EN-DE（约4.5M句对），BPE共享词表约37k；WMT14 EN-FR（36M句对），32k word-piece
* 硬件：8×NVIDIA P100，base训练100k steps约12小时；big 300k steps约3.5天
* 优化器：Adam，(\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9})
* 学习率策略（非常关键）：
  [
  lrate=d_{model}^{-0.5}\cdot \min(step^{-0.5},\ step\cdot warmup^{-1.5})
  ]
  warmup_steps=4000
* 正则：dropout（base (P_{drop}=0.1)）+ label smoothing（(\epsilon_{ls}=0.1)）
* 推理：beam size=4，长度惩罚 (\alpha=0.6)（翻译实验）

---

### 3.3 核心难点解析（更直白易懂）

#### 难点1：为什么要除以 (\sqrt{d_k})？

点积注意力里 logits 是 (q\cdot k)。当 (d_k) 大时，点积的方差随 (d_k) 增大，logits 变大，softmax 更“尖”，梯度更小，训练不稳定。缩放相当于把logits的尺度拉回合理范围。

直觉类比：把相似度当温度，(\sqrt{d_k}) 相当于自动调温，避免“过热（过尖）”。

#### 难点2：多头注意力到底解决了什么？

注意力输出是对 V 的加权平均，单头时容易把多种关系揉在一起（比如同时要对齐主谓、指代、局部短语、长距离依赖）。多头相当于让模型并行学多种“对齐视角/子空间”，再拼起来，减少“平均导致的分辨率损失”。

#### 难点3：没有RNN/CNN，模型怎么知道顺序？

答案是 **位置编码**。它直接把位置信号加到token embedding 上，模型通过注意力在内容与位置共同作用下学习“先后/相对距离”的利用方式。论文选择正弦/余弦的一大原因是希望能外推到更长序列。

#### 难点4：解码器为何必须mask？

训练时 decoder 同时看到整段目标序列，如果不mask，第 (i) 位可以直接注意力看到第 (i+1) 位（未来token），等于作弊；mask 强制只看左边，保证自回归性质。

---

## 4. 实验结果与分析

### 4.1 实验设置（数据集/模型/指标/超参/对比）

**任务与数据**

* WMT14 EN-DE：约4.5M句对，BPE共享37k词表
* WMT14 EN-FR：36M句对，32k word-piece

**模型配置（关键两档）**

* **Transformer base**：(N=6), (d_{model}=512), (d_{ff}=2048), (h=8), (d_k=d_v=64), (P_{drop}=0.1), label smoothing 0.1
* **Transformer big**：在表格中给出big行：(d_{model}=1024), (d_{ff}=4096), (h=16), (P_{drop}=0.3)，训练300k steps

**优化与训练**

* Adam + warmup/inv-sqrt 学习率策略（式(3)）+ warmup=4000
* base：100k steps约12小时；big：300k steps约3.5天（8×P100）

**指标**

* BLEU（主指标），以及训练代价（FLOPs估计）

**对比方法**
ByteNet、GNMT、ConvS2S、MoE、以及它们的ensemble等。

---

### 4.2 实验结果（提升幅度与正面评价）

**机器翻译主结果（Table 2）**

* EN-DE：Transformer(big) BLEU **28.4**，相比当时最强结果（含ensemble）提升 **>2 BLEU**
* EN-FR：Transformer(big) BLEU **41.8**，3.5天8卡训练拿到当时单模型SOTA

**效率**
论文强调 Transformer 更易并行、训练更快，并给出训练成本（FLOPs）对比，整体在更低/可比成本下达到更好BLEU。

**消融/变体（Table 3）带来的结论**

* 单头注意力会明显掉BLEU（约0.9 BLEU差距量级），“头数太多”也会下降：说明多头有“甜点区间”
* 缩小 (d_k) 会伤性能：说明“兼容度计算”并不简单，需要足够维度表达
* dropout 对防过拟合有效：不同dropout设置显著影响PPL/BLEU
* learned positional embedding vs sinusoidal：效果几乎一样（论文最终选sinusoidal为了外推） 

**泛化到句法分析（Parsing，Table 4）**
Transformer 在 WSJ constituency parsing 上也表现很强：

* WSJ only：F1 91.3
* semi-supervised：F1 92.7，超过多种此前方法，仅略低于当时最强RNN Grammar一类模型

---

## 5. 结论

### 5.1 论文的贡献

1. **提出Transformer：首个完全基于注意力的序列转导架构**，替代RNN/CNN成为主干。
2. **并行训练显著更友好**，在机器翻译上以更少训练时间获得更高BLEU。
3. **关键组件体系化**：scaled dot-product attention、multi-head、sinusoidal positional encoding、残差+LayerNorm、label smoothing、warmup学习率等，形成可复用范式 。
4. **展示可解释性线索**：不同attention head学到不同语言/结构行为（论文附录图示）。

### 5.2 论文的限制（哪些方面有问题）

1. **自注意力计算与内存是 (O(n^2))**：序列长度 (n) 很大时成本高（论文在对比中指出复杂度项，并提到可做“局部/受限注意力”来缓解） 。
2. **位置信息注入较“外置”**：位置编码是加法注入，表达相对位置的方式较间接；后续研究大量围绕更强的相对位置建模展开（这点是从论文动机可合理推出：它自己也强调“需要注入位置信息”）。
3. **解码仍是自回归**：训练可并行，但生成（推理）依然逐token生成，串行瓶颈还在（论文在引言/结论也强调“让生成更不串行”是未来方向之一）.

### 5.3 未来方向（可能的发展方向）

论文明确提出的方向包括：

1. **局部/受限注意力**：把注意力限制在邻域 (r)，降低长序列成本，并在未来进一步研究。
2. **扩展到文本之外模态**：把Transformer用于图像、音频、视频等更大输入/输出场景。
3. **让生成更少串行**：减少自回归推理的序列依赖（论文在结论中点名为研究目标之一）。
