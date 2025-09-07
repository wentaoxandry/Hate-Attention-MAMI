#!/bin/bash -u
stage=6			# start from -1 if you need to start from data download
stop_stage=100			# stage at which to stop

datasource=./Sourcedata
dsetdir=./dataset/Memotion3
outdir=./output/Memotion3
cashedir=./CASHE

#conda activate memotion
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    if [ "$(ls -A ${dsetdir})" ]; then
    	echo "Dataset already existed"
    else
    	python3 local/datasets/dataset_memotion.py --sourcedir ${datasource}/Memotion3 \
    						   --savedir ${dsetdir} || exit 1;
        
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Unimodal/memotion_text/taskA/results)" ]; then
    	echo "memotion unimodal model already pretrained"
    else
        for modal in  memotion_text memotion_image; do # 
        python3 local/Memotion/unimodal/trainA.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskA		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
        done                           
    fi      
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Unimodal/memotion_image/taskC/motivation/results)" ]; then
    	echo "memotion unimodal model already pretrained"
    else
        for modal in memotion_image memotion_text; do # 
            for class in humorous sarcastic offensive motivation; do # humorous sarcastic offensive 
                python3 local/Memotion/unimodal/trainC.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskC        \
                                             --classname $class		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
            done
        done                           
    fi      
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ "$(ls -A ${dsetdir}/RMs)" ]; then
    	echo "Signal-based features already extracted"
    else
      python3 local/Memotion/multimodal/DSW_represent/features.py --datadir ${dsetdir} || exit 1;
    fi
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Representation_no_RMs)" ]; then
    	echo "Memotion multimodal DSW and representation fusion already done"
    else
      for modal in Representation DSW; do #DSW
          for RMs in all_RMs text_RMs image_RMs no_RMs; do #  no_RMs
    	      python3 local/Memotion/multimodal/DSW_represent/trainA.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskA        \
                                             --RMs $RMs         \
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;

         done
      done
      
    fi
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Representation_no_RMs/taskC/motivation/results)" ]; then
    	echo "Memotion multimodal DSW and representation fusion already done"
    else
      for modal in Representation DSW; do #DSW
          for RMs in   image_RMs no_RMs text_RMs all_RMs; do #  no_RMs text_RMs all_RMs
              for class in humorous sarcastic offensive motivation; do
    	            python3 local/Memotion/multimodal/DSW_represent/trainC.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskC        \
                                             --RMs $RMs        \
                                             --classname $class		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
               done

         done
      done
      
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-CLIPper/taskA/results)" ]; then
    	echo "memotion Hate-CLIPper model already pretrained"
    else
        modal=Hate-CLIPper
        python3 local/Memotion/multimodal/Hate_CLIPper/trainA.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskA		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;                         
    fi      
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-CLIPper/taskC/motivation/results)" ]; then
    	echo "memotion Hate-CLIPper model already pretrained"
    else
            modal=Hate-CLIPper
            for class in humorous sarcastic offensive motivation; do # humorous sarcastic offensive 
                python3 local/Memotion/multimodal/Hate_CLIPper/trainC.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskC        \
                                             --classname $class		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
            done                         
    fi      
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-CLIPper-cv/taskA/results)" ]; then
    	echo "memotion Hate-CLIPper model already pretrained"
    else
        modal=Hate-CLIPper
        python3 local/Memotion/multimodal/Hate_CLIPper/trainA_cv.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskA		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;                         
    fi      
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate-CLIPper-cv/taskC/motivation/results)" ]; then
    	echo "memotion Hate-CLIPper model already pretrained"
    else
            modal=Hate-CLIPper
            for class in humorous sarcastic offensive motivation; do # humorous sarcastic offensive 
                python3 local/Memotion/multimodal/Hate_CLIPper/trainC_cv.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskC        \
                                             --classname $class		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
            done                         
    fi      
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate_Attention-config2/taskA/results)" ]; then
    	echo "memotion Hate_Attention model already pretrained"
    else
        modal=Hate_Attention
        for config in config1 config2; do
        python3 local/Memotion/multimodal/Hate_Attention/trainA.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskA		\
                                             --config $config   \
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;     
        done                    
    fi      
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate_Attention-config2/taskC/motivation/results)" ]; then
    	echo "memotion Hate_Attention model already pretrained"
    else
            modal=Hate_Attention
            for config in config1 config2; do
            for class in humorous sarcastic offensive motivation; do # humorous sarcastic offensive 
                python3 local/Memotion/multimodal/Hate_Attention/trainC.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskC        \
                                             --config $config   \
                                             --classname $class		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
            done    
            done                     
    fi      
fi



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate_Attention-config2-cv/taskA/results)" ]; then
    	echo "memotion Hate_Attention model already pretrained"
    else
        modal=Hate_Attention
        for config in config1 config2; do
        python3 local/Memotion/multimodal/Hate_Attention/trainA_cv.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskA		\
                                             --config $config   \
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;     
        done                    
    fi      
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    if [ "$(ls -A ${outdir}/Multimodal/Hate_Attention-config2-cv/taskC/motivation/results)" ]; then
    	echo "memotion Hate_Attention model already pretrained"
    else
            modal=Hate_Attention
            for config in config1 config2; do
            for class in humorous sarcastic offensive motivation; do # humorous sarcastic offensive 
                python3 local/Memotion/multimodal/Hate_Attention/trainC_cv.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskC        \
                                             --config $config   \
                                             --classname $class		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
            done    
            done                     
    fi      
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    if [ "$(ls -A ${outdir}/Unimodal/memotion_text_fine-tune/taskA/results)" ]; then
    	echo "memotion unimodal model already pretrained"
    else
        for modal in  memotion_text memotion_image; do # 
        python3 local/Memotion/unimodal/trainA_fine_tune.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskA		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
        done                           
    fi      
fi



if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    if [ "$(ls -A ${outdir}/Unimodal/memotion_image_fine-tune/taskC/motivation/results)" ]; then
    	echo "memotion unimodal model already pretrained"
    else
        for modal in memotion_image; do # 
            for class in offensive motivation; do # humorous sarcastic offensive 
                python3 local/Memotion/unimodal/trainC_fine_tune.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskC        \
                                             --classname $class		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
            done
        done                           
    fi      
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    if [ "$(ls -A ${outdir}/Unimodal/memotion_text_fine-tune/taskC/motivation/results)" ]; then
    	echo "memotion unimodal model already pretrained"
    else
        for modal in memotion_text; do # 
            for class in humorous sarcastic offensive motivation; do # humorous sarcastic offensive 
                python3 local/Memotion/unimodal/trainC_fine_tune.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskC        \
                                             --classname $class		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
            done
        done                           
    fi      
fi


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    if [ "$(ls -A ${outdir}/Unimodal/memotion_image_fine-tune/taskC/motivation/results)" ]; then
    	echo "memotion unimodal model already pretrained"
    else
        for modal in memotion_image memotion_text; do # 
            for class in humorous sarcastic offensive motivation; do # humorous sarcastic offensive 
                python3 local/Memotion/unimodal/trainC_fine_tune.py --datadir ${dsetdir} \
                                             --modal $modal        \
                                             --task taskC        \
                                             --classname $class		\
                                             --savedir ${outdir}  \
                                             --cashedir ${cashedir}|| exit 1;
            done
        done                           
    fi      
fi
exit 0


