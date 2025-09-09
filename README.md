# Toxic memes detection

In this project, we propose the **Hate-Attention** model (Figure 1) for the task of misogynous meme detection. We compare our approach with the Hate-CLIPper model [1], which was one of the top-performing methods on the Hateful Memes dataset[2]. In addition, we evaluate state-of-the-art LLMs, including a fine-tuned Tiny-LLaVA model [3], as well as zero-shot and few-shot settings of the ChatGPT-4.o model [4]. All models are evaluated on the [MAMI-22 dataset](https://www.kaggle.com/datasets/chukwuebukaanulunko/multimodal-misogyny-detection-mami-2022?select=validation.tsv) [5].

To the best of our knowledge, the proposed **Hate-Attention-large** model achieves state-of-the-art performance on the MAMI-22 Task B.

![Hate-Attention Model](./image/model.png)  
*Figure 1: Hate-Attention Model Architecture*  

---

## MAMI-22 dataset [5]
The MAMI-22 dataset contains two tasks:
- **Task A**: Identify misogynous memes (binary classification).
- **Task B**: Classify misogynous memes into four potentially overlapping categories: *Shaming*, *Stereotype*, *Objectification*, and *Violence* (multi-label classification).

*We use a post-processing approach to derive Task A results from Task B predictions, simplifying the problem by addressing both tasks with a single model.*


---

## How to Run

1. Create a folder named `./Sourcedata` in the project root directory.  
2. Download the **MAMI-22 dataset** and place it inside the `./Sourcedata` folder.  
3. Run the training script:  
```bash
sh run_MAMI.sh
```

### Stages

- **Stage 0: Dataset preparation** <br>
  Extract text, image paths, and ground-truth labels from the dataset and save them in JSON files for the train, validation, and test sets.

- **Stage 1: Train unimodal models** <br> 
  Train unimodal models for **text** and **image** modalities, based on the pre-trained CLIP encoders [6]:  
  `openai/clip-vit-large-patch14`

- **Stage 2: Train Hate-Attention model** <br>
  We propose a **Hate-Attention model** (Figure 1), built on top of a pre-trained CLIP backbone. The Hate-Attention variants differ in the configuration of the multi-head attention block:

  | Setting | Tiny | Base | Large |
  |---------|------|------|-------|
  | Attention Dimension ($d$) | 768 | 768 | 1024 |
  | Number of Attention Blocks | 0 | 1 | 2 |
  | Number of Attention Heads | – | 16 | 8 |
  | Trainable Parameters | 10,830,340 | 21,858,308 | 43,834,372 | 

  **The Hate-Attention-tiny model is a variant of Hate-CLIPper and will be run in Stage 4.**

- **Stage 3: Cross-validation of the Hate-Attention model** <br>
  We conduct **10-fold cross-validation** for the Hate-Attention model to ensure robust performance evaluation.

- **Stage 4: Train Hate-CLIPper model** <br>
  Train the Hate-CLIPper (Hate-Attention-tiny) model

- **Stage 5: Cross-validation of the Hate-CLIPper model** <br>
  We conduct **10-fold cross-validation** for the Hate-CLIPper model to ensure robust performance evaluation.

- **Stage 6: Train Tiny-LLaVA model** <br>
  Fine-tune the pre-trained **TinyLLaVA** model [3]

- **Stage 7: Non-optimized ChatGPT 4.o model** <br>
  Zero-shot and few-shot **ChatGPT-4.o** model [4] 

- **Stage 8: Analysis** <br>
  We conduct a detailed performance analysis across all models. Additionally, we compute the **Expected Calibration Error (ECE)** [7] and visualize results with **reliability diagrams** ([code reference](https://github.com/hollance/reliability-diagrams)).   

---


## References
[1] Gokul et al. *Hate-CLIPper: Multimodal hateful meme classification based on cross-modal interaction of CLIP features*. arXiv preprint arXiv:2210.05916, 2022. 
[2] Kiela et al. *The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes*. NeurIPS 2020.   
[3] Zhou et al. *TinyLLaVA: A Framework of Small-Scale Large Multimodal Models*. arXiv:2402.14289, 2024.  
[4] OpenAI. *ChatGPT-4*. https://openai.com/chatgpt. Accessed Jan 19, 2025. 
[5] Fersini et al. *SemEval-2022 Task 5: Multimedia Automatic Misogyny Identification*. SemEval 2022.  
[6] Radford et al. *Learning Transferable Visual Models from Natural Language Supervision*. ICML 2021.  
[7] Guo et al. *On Calibration of Modern Neural Networks*. ICML 2017.  