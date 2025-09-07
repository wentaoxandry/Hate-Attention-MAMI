#!/bin/bash -u
# -----------------------------------------------------------------------------
# Pipeline driver for data prep, (uni/multi)modal training, cross-validation, and analysis.
# Stages:
#   0 - Prepare datasets (generate train/eval/test JSON)
#   1 - Train unimodal baselines (text-only, image-only) for taskA/taskB
#   2 - Train Hate-Attention model (taskA/taskB)
#   3 - Cross-validate Hate-Attention (taskA/taskB)
#   4 - Train Hate-CLIPper baseline (taskA/taskB)
#   5 - Cross-validate Hate-CLIPper (taskA/taskB)
#   8 - Run statistical tests, modality analysis, ECE for Unimodal/Multimodal
# -----------------------------------------------------------------------------


stage=0                  # (int) Start stage index. Use 0 to include dataset prep.
stop_stage=100           # (int) Stop stage index. Larger value runs all stages.

datasource=./../MAMI_test/Sourcedata #./Sourcedata # (path) Root directory containing downloaded raw data.
metadir=./meta_info/MAMI # (path) Directory to save the meta information for reproduction
dsetdir=./../MAMI_test/dataset/MAMI #./dataset/MAMI  # (path) Directory to save processed dataset (JSON files).
outdir=./../MAMI_test/output/MAMI #./output/MAMI     # (path) Directory to save trained models and results.
cashedir=./../CASHE #./CASHE         # (path) Directory for cached/downloaded pretrained models.


# --- Stage 0: dataset preparation ------------------------------------------------
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ "$(ls -A ${dsetdir})" ]; then
    	echo "Dataset already existed"
    else
        # Generate JSON data for train/eval/test with text, image path, and labels.
    	python3 local/datasets/dataset_mami.py  --sourcedir ${datasource}/MAMI      \
    						                    --savedir ${dsetdir}                \
                                                --metadir ${metadir} || exit 1;
        
    fi
fi

# --- Stage 1: unimodal pretraining (image-only, text-only) ----------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Unimodal/mami_image/taskB/results)" ]; then
    	echo "mami unimodal model already pretrained"
    else
        # Train text-only and image-only models for both tasks.
        for modal in mami_image mami_text; do # 
            for task in taskA taskB; do #
                python3 local/MAMI/unimodal/train.py --datadir ${dsetdir}           \
                                                     --modal $modal                 \
                                                     --task $task                   \
                                                     --savedir ${outdir}            \
                                                     --cashedir ${cashedir} || exit 1;
            done
        done                             
    fi      
fi

# --- Stage 2: Hate-Attention training ------------------------------------------
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-Attention-config2/taskB/results)" ]; then
    	echo "Hate-Attention model already trained"
    else
        # Train proposed Hate-Attention model (multimodal) for both tasks.
        modal=Hate-Attention
        for task in  taskA taskB ; do 
            python3 local/MAMI/multimodal/Hate_Attention/train.py --datadir ${dsetdir}  \
                                                                  --modal $modal        \
                                                                  --task $task          \
                                                                  --savedir ${outdir}   \
                                                                  --cashedir ${cashedir} || exit 1;
        done
    fi      
fi

# --- Stage 3: Hate-Attention cross-validation -----------------------------------
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-Attention-cv-config2/taskB/9)" ]; then
    	echo "Hate-Attention model cross validation already trained"
    else
        # Cross-validate Hate-Attention model for both tasks.
        modal=Hate-Attention
        for task in  taskA taskB  ; do 
            python3 local/MAMI/multimodal/Hate_Attention/train_cv.py --datadir ${dsetdir}   \
                                                                     --metadir ${metadir}   \
                                                                     --modal $modal         \
                                                                     --task $task           \
                                                                     --savedir ${outdir}    \
                                                                     --cashedir ${cashedir} || exit 1;
        done
    fi      
fi

# --- Stage 4: Hate-CLIPper training --------------------------------------------
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-CLIPper/taskB/results)" ]; then
    	echo "Hate-CLIPper model already trained"
    else
        # Train Hate-CLIPper baseline (multimodal) for both tasks.
        modal=Hate-CLIPper
        for task in  taskA taskB ; do 
            python3 local/MAMI/multimodal/Hate_CLIPper/train.py --datadir ${dsetdir}        \
                                                                --modal $modal              \
                                                                --task $task                \
                                                                --savedir ${outdir}         \
                                                                --cashedir ${cashedir} || exit 1;
        done
    fi      
fi

# --- Stage 5: Hate-CLIPper cross-validation ------------------------------------
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-CLIPper-cv/taskB/9)" ]; then
    	echo "Hate-CLIPper model cross validation already trained"
    else
        # Cross-validate Hate-CLIPper model for both tasks.
        modal=Hate-CLIPper
        for task in  taskA taskB ; do 
        python3 local/MAMI/multimodal/Hate_CLIPper/train_cv.py --datadir ${dsetdir}     \
                                                               --metadir ${metadir}     \
                                                               --modal $modal           \
                                                               --task $task             \
                                                               --savedir ${outdir}      \
                                                               --cashedir ${cashedir} || exit 1;
        done
    fi      
fi

# --- Stage 6: Fine-tune Tiny-LLaVA model --------------------------------------
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Tiny-LLaVA/taskB/results)" ]; then
    	echo "Tiny-LLaVA model already trained"
    else
        # Train Tiny-LLaVA baseline (multimodal) for both tasks.
        modal=Tiny-LLaVA
        for task in  taskA taskB ; do 
            python3 local/MAMI/multimodal/TinyLLaVA/train.py --datadir ${dsetdir}        \
                                                             --modal $modal              \
                                                             --task $task                \
                                                             --savedir ${outdir} || exit 1;
        done
    fi      
fi

# --- Stage 7: ChatGPT 4.o model -------------------------------------------------
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/ChatGPT/few_shot)" ]; then
    	echo "ChatGPT model already run"
    else
        # zero-shot and few shot ChatGPT 4.o model for both tasks.
        modal=ChatGPT
        python3 local/MAMI/multimodal/ChatGPT_API/GPT-Captioning-zeroshot.py --sourcedir ${datasource}/MAMI         \
                                                                             --modal $modal                         \
                                                                             --savedir ${outdir} || exit 1;
        python3 local/MAMI/multimodal/ChatGPT_API/GPT-Captioning-fewshot.py --datadir ${dsetdir}                    \
                                                                             --modal $modal                         \
                                                                             --savedir ${outdir} || exit 1;                                                                    
    fi      
fi

# --- Stage 8: statistical tests, analysis, calibration --------------------------
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    # get performance of each model for each category
    python3 local/MAMI/modal_analyse.py --resultsdir ${outdir}          \
                                        --modaltype Unimodal || exit 1;   
    # plot reliability diagram for each model                                        
    python3 local/MAMI/get_ece.py --resultsdir ${outdir}                \
                                  --modaltype Unimodal || exit 1;  
    python3 local/MAMI/modal_analyse.py --resultsdir ${outdir}          \
                                        --modaltype Multimodal || exit 1;   
    python3 local/MAMI/get_ece.py --resultsdir ${outdir}                \
                                  --modaltype Multimodal || exit 1;                                         
fi
