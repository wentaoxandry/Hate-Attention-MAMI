# Toxic Memes Detection

This project focuses on the **detection of misogynous memes**.  
We evaluate models using the [MAMI-22 dataset](https://www.kaggle.com/datasets/chukwuebukaanulunko/multimodal-misogyny-detection-mami-2022?select=validation.tsv) [1].  

---

## 1. Unimodal Models
We first train unimodal models for **text** and **image** modalities, based on the pre-trained CLIP encoders [2]:  
`openai/clip-vit-large-patch14`  

---

## 2. Multimodal Models

### 2.1 Hate-Attention Model
We propose a **Hate-Attention model**, built on top of a pre-trained CLIP backbone.  

![Hate-Attention Model](./image/model.png)  
*Figure 1: Hate-Attention Model Architecture*  

The Hate-Attention variants differ in the configuration of the multi-head attention block (Figure 1):  

| Setting | Tiny | Base | Large |
|---------|------|------|-------|
| Attention Dimension ($d$) | 768 | 768 | 1024 |
| Number of Attention Blocks | 0 | 1 | 2 |
| Number of Attention Heads | – | 16 | 8 |
| Trainable Parameters | 10,830,340 | 21,858,308 | 43,834,372 |

- The **Hate-Attention-tiny** model is a variant of **Hate-CLIPper** [3], which was a top-performing approach on the Hateful Memes dataset.  

---

### 2.2 State-of-the-Art LLM Baselines
We also compare Hate-Attention against recent large language models:  
- **TinyLLaVA** [4]  
- **ChatGPT-4.0** [5], under zero-shot and few-shot settings  

---

## 3. Analysis
We conduct a detailed performance analysis across all models.  
Additionally, we compute the **Expected Calibration Error (ECE)** [6] and visualize results with **reliability diagrams** ([code reference](https://github.com/hollance/reliability-diagrams)).  

---

## 4. How to Run

1. Download the **MAMI-22 dataset** and save it under the folder: `./Sourcedata`.
2. Run the training script:  
```bash
sh run_MAMI.sh
```

---

## References
[1] Fersini et al. *SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification*. SemEval 2022.  
[2] Radford et al. *Learning Transferable Visual Models from Natural Language Supervision*. ICML 2021.  
[3] Kiela et al. *The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes*. NeurIPS 2020.  
[4] Zhou et al. *TinyLLaVA: A Framework of Small-Scale Large Multimodal Models*. arXiv:2402.14289, 2024.  
[5] OpenAI. *ChatGPT-4*. https://openai.com/chatgpt. Accessed Jan 19, 2025.  
[6] Guo et al. *On Calibration of Modern Neural Networks*. ICML 2017.  