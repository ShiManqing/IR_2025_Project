# IR 2025 Project — WebQuestions 检索模型对比实验

本项目是《现代信息检索》课程大作业，实现并对比不同**检索模型（Retriever）**在 **WebQuestions (WebQ)** 数据集上的性能，评测指标为 **Hit@K**。

当前已完成并对比的模型包括：

- ✅ **TF-IDF Retriever**  
- ✅ **BM25 Retriever**

所有模型均 **复用老师提供的评测接口**（`retriever_utils.py`、`cal_hit_multi.py`），不修改原有评测逻辑，生成的结果文件格式保持一致，可直接用于课程作业提交。

---

## 一、项目目录结构说明

```text
IR_2025_Project/
├── corpus/
│   └── wiki_webq_corpus.tsv        # Wikipedia 语料（id, text, title）
│
├── datas/
│   ├── webq-train.json             # WebQuestions 训练集
│   ├── webq-dev.json               # WebQuestions 验证集
│   ├── webq-test.csv               # 测试集（question + answers）
│   └── webq-test.txt
│
├── util/
│   ├── retriever_utils.py          # 【老师提供】语料加载 + 评测接口
│   ├── tfidf_retriever.py          # TF-IDF 检索模型实现
│   └── bm25_retriever.py           # BM25 检索模型实现
│
├── DPR/
│   └── dpr/
│       └── data/
│           └── qa_validation.py    # 【老师提供】答案匹配与命中判定逻辑
│
├── output/
│   ├── result_str.pkl              # 检索结果文件（供评测脚本读取）
│   └── *.pkl / *.csv               # 各模型索引文件与评测输出结果
│
├── run_retrieval.py                # 检索主入口脚本（支持多模型切换）
├── cal_hit_multi.py                # 【老师提供】Hit@K 评测脚本
└── README.md
```

---

## 二、已实现的检索模型简介

### 1️⃣ TF-IDF Retriever

- 基于经典 **TF-IDF 向量空间模型**
- 使用词项权重计算查询与文档之间的相似度
- 优点：实现简单、构建索引和检索速度快
- 缺点：仅基于词面匹配，对同义词和语义相似不敏感

实现文件：

```text
util/tfidf_retriever.py
```

---

### 2️⃣ BM25 Retriever

- 基于概率检索模型 **BM25 (Okapi)**
- 在 TF-IDF 基础上引入：
  - 文档长度归一化
  - 非线性词频饱和机制
- 在问答检索任务中通常比 TF-IDF 具有更好的 Hit@K 表现

实现文件：

```text
util/bm25_retriever.py
```

> ⚠️ 注意：BM25 检索结果中输出的 `doc_id` 与语料 `wiki_webq_corpus.tsv` 中第一列 **严格保持字符串一致**，以确保与老师提供的评测脚本完全兼容。

---

## 三、运行说明

### 1️⃣ 环境准备

推荐使用 Conda 创建独立环境：

```bash
conda create -n dpr python=3.9
conda activate dpr
pip install -r requirements.txt
```

> 本项目仅使用 CPU 即可运行，无需 GPU。

---

### 2️⃣ 运行 TF-IDF 检索并评测

```bash
python run_retrieval.py --model tfidf --topk 100
python cal_hit_multi.py
```

---

### 3️⃣ 运行 BM25 检索并评测

```bash
python run_retrieval.py --model bm25 --topk 100
python cal_hit_multi.py
```

说明：

- 所有模型都会生成统一格式的检索结果文件：
  ```text
  output/result_str.pkl
  ```
- `cal_hit_multi.py` 会自动读取该文件并计算 Hit@K
- 不同模型使用不同的索引文件（如 `tfidf_index.pkl`、`bm25_index.pkl`），避免相互覆盖

---

## 四、评测指标说明（Hit@K）

**Hit@K** 定义如下：

> 对于每个问题，如果其任一正确答案在 Top-K 检索结果中被命中，则记为 1，否则为 0；  
> Hit@K 为所有问题命中比例的平均值。

评测逻辑完全由老师提供的脚本完成：

```text
util/retriever_utils.py
DPR/dpr/data/qa_validation.py
```

---

## 五、实验结果示例

以下为示例结果（Top-100 检索），具体数值以实际运行输出为准：

| 模型   | Hit@1 | Hit@10 | Hit@20 | Hit@50 | Hit@100 |
|--------|-------|--------|--------|--------|----------|
| TF-IDF | 0.16 | 0.395 | 0.515 | 0.62  | 0.73    |
| BM25   | 0.215 | 0.53 | 0.61   | 0.7   | 0.755   |

可以观察到：

- BM25 在所有 K 值上均优于 TF-IDF
- 在 Top-10 / Top-20 等常用检索区间，BM25 提升尤为明显

---

## 六、扩展说明

- 本项目严格遵循课程要求，**复用老师提供的评测接口**，保证不同模型结果公平可比
- 当前实现的均为稀疏检索模型（Sparse Retrieval）
- 在不改变评测接口的前提下，可进一步扩展：
  - FAISS + LSA
  - Dense Retriever（Sentence-BERT + FAISS）
  - Hybrid Retriever（BM25 + Dense）

---

## 七、作者说明

- 本项目用于《现代信息检索》课程大作业
- 所有实现均用于教学与实验目的
- 可在此基础上继续扩展与优化

