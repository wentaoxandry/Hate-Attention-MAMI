#!/bin/bash -u
stage=0			        # stage at which to beginning, start from 0 if you need 
                        # to process dataset
stop_stage=100			# stage at which to stop

datasource=./Sourcedata # Path save downloaded raw data
dsetdir=./dataset/MAMI  # Path save processed dataset (generate JSON file for training)
outdir=./output/MAMI    # Path save trained model and the results
cashedir=./CASHE        # Path save downloaded pre-trained language models


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ "$(ls -A ${dsetdir})" ]; then
    	echo "Dataset already existed"
    else
        # This script generate JSON data for train, eval and test set. Each JSON file 
        # contains text, image path and the ground truth label of the sample
    	python3 local/datasets/dataset_mami.py  --sourcedir ${datasource}/MAMI      \
    						                    --savedir ${dsetdir} || exit 1;
        
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Unimodal/mami_image/taskB/results)" ]; then
    	echo "mami unimodal model already pretrained"
    else
        # Here we train the text and image-only models
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

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-Attention-config2/taskB/results)" ]; then
    	echo "Hate-Attention model already trained"
    else
        # This python script is for our proposed Hate-Attention model
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
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-Attention-cv-config2/taskB/9)" ]; then
    	echo "Hate-Attention model cross validation already trained"
    else
        # This python script is for Hate-Attention model cross-validation
        modal=Hate-Attention
        for task in  taskA taskB  ; do 
            python3 local/MAMI/multimodal/Hate_Attention/train_cv.py --datadir ${dsetdir}   \
                                                                     --modal $modal         \
                                                                     --task $task           \
                                                                     --savedir ${outdir}    \
                                                                     --cashedir ${cashedir} || exit 1;
        done
    fi      
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-CLIPper/taskB/results)" ]; then
    	echo "Hate-CLIPper model already trained"
    else
        # This python script is for Hate-CLIPper baseline model
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

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-CLIPper-cv/taskB/9)" ]; then
    	echo "Hate-CLIPper model cross validation already trained"
    else
        # This python script is for Hate-CLIPper model cross-validation
        modal=Hate-CLIPper
        for task in  taskA taskB ; do 
        python3 local/MAMI/multimodal/Hate_CLIPper/train_cv.py --datadir ${dsetdir}     \
                                                               --modal $modal           \
                                                               --task $task             \
                                                               --savedir ${outdir}      \
                                                               --cashedir ${cashedir} || exit 1;
        done
    fi      
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    python3 local/MAMI/significant_test.py --resultsdir ${outdir}\
                                                --modaltype Unimodal || exit 1;
    python3 local/MAMI/modal_analyse.py --resultsdir ${outdir}\
                                                --modaltype Unimodal || exit 1;   
    python3 local/MAMI/get_ece.py --resultsdir ${outdir}\
                                                --modaltype Unimodal || exit 1;  
    python3 local/MAMI/significant_test.py --resultsdir ${outdir}\
                                                --modaltype Multimodal || exit 1;
    python3 local/MAMI/modal_analyse.py --resultsdir ${outdir}\
                                                --modaltype Multimodal || exit 1;   
    python3 local/MAMI/get_ece.py --resultsdir ${outdir}\
                                                --modaltype Multimodal || exit 1;                                         
fi
