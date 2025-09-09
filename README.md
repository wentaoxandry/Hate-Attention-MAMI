# Toxic memes detection

In this project, we propose the **Hate-Attention** model (Figure 1) for the task of misogynous meme detection. We compare our approach with the Hate-CLIPper model [1], which was one of the top-performing methods on the Hateful Memes dataset. In addition, we evaluate state-of-the-art LLMs, including a fine-tuned Tiny-LLaVA model, as well as zero-shot and few-shot settings of the ChatGPT-4.0 model. All models are evaluated on the [MAMI-22 dataset](https://www.kaggle.com/datasets/chukwuebukaanulunko/multimodal-misogyny-detection-mami-2022?select=validation.tsv) [2].

To the best of our knowledge, the proposed **Hate-Attention-large** model achieves state-of-the-art performance on the MAMI-22 Task B.

![Hate-Attention Model](./image/model.png)  
*Figure 1: Hate-Attention Model Architecture*  

---

## MAMI-22 dataset
The MAMI-22 dataset contains two tasks:
- **Task A**: Identify misogynous memes (binary classification).
- **Task B**: Classify misogynous memes into four potentially overlapping categories: *Shaming*, *Stereotype*, *Objectification*, and *Violence* (multi-label classification).

*We use a post-processing approach to derive Task A results from Task B predictions, simplifying the problem by addressing both tasks with a single model.*


---

## How to Run

1. Download the **MAMI-22 dataset** and save it under the folder: `./Sourcedata`.
2. Run the training script:  
```bash
sh run_MAMI.sh
```

### Stages

- **Stage 0: Dataset preparation**  
  Extract text, image paths, and ground-truth labels from the dataset and save them in JSON files for the train, validation, and test sets.

- **Stage 1: Train unimodal models**  
  Train unimodal models for **text** and **image** modalities, based on the pre-trained CLIP encoders [2]:  
  `openai/clip-vit-large-patch14`

- **Stage 2: Train Hate-Attention model** <br>
  We propose a **Hate-Attention model** (Figure 1), built on top of a pre-trained CLIP backbone. The Hate-Attention variants differ in the configuration of the multi-head attention block:

  | Setting | Tiny | Base | Large |
  |---------|------|------|-------|
  | Attention Dimension ($d$) | 768 | 768 | 1024 |
  | Number of Attention Blocks | 0 | 1 | 2 |
  | Number of Attention Heads | – | 16 | 8 |
  | Trainable Parameters | 10,830,340 | 21,858,308 | 43,834,372 | 

  **The Hate-Attention-tiny model is a variant of Hate-CLIPper.**

- **Stage 3: Cross-validation of the Hate-Attention model**

---




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