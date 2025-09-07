import os, sys, json, argparse
import numpy as np
import scipy
from sklearn.metrics import f1_score
import numpy as np
import logging

from sklearn import metrics


def get_taskA_score(modeldir, modelname):
    datadir = os.path.join(modeldir, "taskA", "results")
    for i in os.listdir(datadir):
        if 'test' in i:
            targetfile = i
    with open(os.path.join(datadir, targetfile), encoding="utf8") as json_file:
        resultsdict = json.load(json_file)
    label = []
    predict = []
    for key in list(resultsdict.keys()):
        label.append(int(resultsdict[key]['label']))
        predict.append(int(resultsdict[key]['predict']))
    score = f1_score(np.asarray(label), np.asarray(predict), average='weighted')
    return score




def get_taskBC_score(modeldir, modelname, testdict):
    maindir = os.path.join(modeldir, "taskC")
    alltaskB = []
    alltaskC = []
    catescore = {}
    for cate in os.listdir(maindir):
        catescore.update({cate: {}})
        datadir = os.path.join(modeldir, "taskC", cate, "results")
        for i in os.listdir(datadir):
            if 'test' in i:
                targetfile = i
        with open(os.path.join(datadir, targetfile), encoding="utf8") as json_file:
            resultsdict = json.load(json_file)
        taskClabel = []
        taskCpredict = []
        taskBlabel = []
        taskBpredict = []
        for key in list(resultsdict.keys()):
            taskCpred = int(resultsdict[key]['predict'])
            taskClab = int(resultsdict[key]['label'])
            if taskCpred != 0:
                taskBpred = 1
            else:
                taskBpred = 0
            taskBlab = int(testdict[key]['taskB'][cate])
            taskClabel.append(taskClab)
            taskCpredict.append(taskCpred)
            taskBlabel.append(taskBlab)
            taskBpredict.append(taskBpred)

        if cate == 'sarcastic':
            for i in range(len(taskCpredict)):
                if taskCpredict[i] == 1:
                    taskCpredict[i] = 3
                elif taskCpredict[i] == 2:
                    taskCpredict[i] = 3
        taskBscore = f1_score(np.asarray(taskBlabel), np.asarray(taskBpredict), average='weighted')
        catescore[cate].update({'taskB': taskBscore})
        taskCscore = f1_score(np.asarray(taskClabel), np.asarray(taskCpredict), average='weighted')
        catescore[cate].update({'taskC': taskCscore})

        alltaskB.append(taskBscore)
        alltaskC.append(taskCscore)
    meantaskB = sum(alltaskB) / len(alltaskB)
    meantaskC = sum(alltaskC) / len(alltaskC)
    return catescore, meantaskB, meantaskC

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--resultsdir', default='./output/Memotion3', type=str,
                        help='Dir saves the datasource information')
    parser.add_argument('--modaltype', default='Multimodal', type=str,
                        help='Dir saves the datasource information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    resultsdir = args.resultsdir
    modaltype = args.modaltype
    logging.basicConfig(filename=os.path.join(resultsdir, 'F1 score.log'), level=logging.INFO)
    logging.info(f'Unimodal F1 scores')

    if modaltype == 'Unimodal':
        modals = ['Challenge-image-0', 'Challenge-image-1', 'Challenge-text-0', 'Challenge-text-1', 'memotion_image', 
                  'memotion_text', 'memotion_image_fine-tune', 'memotion_text_fine-tune']
    else:
        modals = ['DSW_text_RMs', 'DSW_image_RMs', 'DSW_all_RMs', 'DSW_no_RMs',
                  'Representation_text_RMs', 'Representation_image_RMs', 'Representation_all_RMs', 'Representation_no_RMs',
                  'Hate-CLIPper', 'Hate_Attention-config1', 'Hate_Attention-config2', 'Challenge-multi', 'Challenge-oscar', 'CLIP_m', 'CLIP_m_right',
                    'ensemble_0', 'ensemble_1', 'ChatGPT', "ChatGPT_zeroshot", 'TinyLLaVA']
    modalsdir = [os.path.join(resultsdir, modaltype, i) for i in modals]
    with open(os.path.join('./dataset/Memotion3/test_en_labels.json'), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    for modal, modaldir in zip(modals, modalsdir):
        for task in ["taskA", "taskC"]:
            if task == "taskA":
                allscore = get_taskA_score(modaldir, modal)
                logging.info(f'{modal} {task} F1 score is {allscore}')
            else:
                catescore, meantaskB, meantaskC = get_taskBC_score(modaldir, modal, testdict)
                for i in list(catescore.keys()):
                    logging.info(f'{modal} TaskB {i} F1 score is {catescore[i]["taskB"]}')
                logging.info(f'{modal} TaskB average F1 score is {meantaskB}')
                for i in list(catescore.keys()):
                    logging.info(f'{modal} TaskC {i} F1 score is {catescore[i]["taskC"]}')
                logging.info(f'{modal} TaskC average F1 score is {meantaskC}')







