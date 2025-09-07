# Hateful Memes

This project focuses on the **detection of misogynous memes**. We evaluate all models using the [MAMI-22 dataset](https://aclanthology.org/2022.semeval-1.69/) [1].  

---

## Methodology

### 1. Baseline Models
We first run unimodal models for both text and image modalities, based on the pre-trained CLIP encoders [2] (`openai/clip-vit-large-patch14`).  

### 2. Hate-Attention Model
We then propose and optimize a **Hate-Attention model**, built on top of a pre-trained CLIP model.  

The model architecture is shown in:  
`./image/model.pdf` *(add figure here with a caption)*  

#### Hate-Attention Configurations
| Setting | Tiny | Base | Large |
|---------|------|------|-------|
| Attention Dimension ($d$) | 768 | 768 | 1024 |
| Number of Attention Blocks | 0 | 1 | 2 |
| Number of Attention Heads | - | 16 | 8 |
| Trainable Parameters | 10,830,340 | 21,858,308 | 43,834,372 |

---

## Comparisons
We compare the Hate-Attention variants with several strong baselines:
- **Hate-CLIPper** (top-performing model on the Hateful Memes dataset) [3]  
- **TinyLLaVA** [4]  
- **ChatGPT-4.0** [5], under zero-shot and few-shot settings  

---

## Analysis
Finally, we analyze and summarize the performance of all models.  
We also compute the **Expected Calibration Error (ECE)** [6] and plot **reliability diagrams** ([code reference](https://github.com/hollance/reliability-diagrams)).  

---

## References
[1] Fersini et al. *SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification*. SemEval 2022.  
[2] Radford et al. *Learning Transferable Visual Models from Natural Language Supervision*. ICML 2021.  
[3] Kiela et al. *The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes*. NeurIPS 2020.  
[4] Zhou et al. *TinyLLaVA: A Framework of Small-Scale Large Multimodal Models*. arXiv:2402.14289, 2024.  
[5] OpenAI. *ChatGPT-4*. https://openai.com/chatgpt. Accessed Jan 19, 2025.  
[6] Guo et al. *On Calibration of Modern Neural Networks*. ICML 2017.  
