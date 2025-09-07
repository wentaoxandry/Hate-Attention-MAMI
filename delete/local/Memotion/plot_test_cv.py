import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import f1_score
import json
from sklearn import metrics
import seaborn as sns

def get_taskA_score(modeldir):

    with open(modeldir, encoding="utf8") as json_file:
        resultsdict = json.load(json_file)
    label = []
    predict = []
    for key in list(resultsdict.keys()):
        label.append(int(resultsdict[key]['label']))
        predict.append(int(resultsdict[key]['predict']))
    score = f1_score(np.asarray(label), np.asarray(predict), average='weighted')
    return score




def get_taskBC_score(modeldir, testdict, cvi):
    maindir = modeldir
    alltaskB = []
    alltaskC = []
    catescore = {}
    for cate in os.listdir(maindir):
        catescore.update({cate: {}})
        datadir = os.path.join(modeldir, cate, str(cvi), "results")
        for i in os.listdir(datadir):
            if 'test_' in i:
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


def get_data_norm(modaldir, modal, testdict):
    scoreA = []
    scoreB = []
    scoreC = []
    for i in range(5):
        subresultsAdir = os.path.join(modaldir, 'taskA', str(i))
        resultsAdir = os.path.join(subresultsAdir, 'results')
        fileAlist = os.listdir(resultsAdir)
        fileA = [x for x in fileAlist if 'test' in x][0]
        resultsfileAdir = os.path.join(resultsAdir, fileA)
        allscore = get_taskA_score(resultsfileAdir)
        scoreA.append(allscore)
        
    for i in range(5):
        subresultsBCdir = os.path.join(modaldir, 'taskC')
        _, meantaskB, meantaskC = get_taskBC_score(subresultsBCdir, testdict, i)
        scoreB.append(meantaskB)
        scoreC.append(meantaskC)

    return scoreA, scoreB, scoreC
def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./output/Memotion3', type=str, help='Dir saves the datasource information')
    parser.add_argument('--sourcedir', default='./dataset/Memotion3', type=str, help='Dir saves the datasource information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    sourcedir = args.sourcedir

    with open(os.path.join('./dataset/Memotion3/test_en_labels.json'), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    models = ['Hate-CLIPper-cv', 'Hate_Attention-config1-cv', 'Hate_Attention-config2-cv']
    modalsdir = [os.path.join(datadir, 'Multimodal', i) for i in models]

    allscores = {}
    for modal, modaldir in zip(models, modalsdir):
        allscores.update({modal: {}})
        scoreA, scoreB, scoreC = get_data_norm(modaldir, modal, testdict)
        allscores[modal].update({'taskA': scoreA})
        allscores[modal].update({'taskB': scoreB})
        allscores[modal].update({'taskC': scoreC})
    
    fontsize = 18
    for type in ['A', 'B', 'C']:
        for model in models:
            score = allscores[model]['task' + type]
            cvs = list(range(5))
            cvs = [int(k + 1) for k in cvs]
            if model == 'Hate-CLIPper-cv':
                model = 'Hate-CLIPper'
            elif model == 'Hate_Attention-config1-cv':
                model = 'Hate-Attention-config1'
            elif model == 'Hate_Attention-config2-cv':
                model = 'Hate-Attention-config2'
            plt.plot(cvs, score, label=model)
         
            # Adding title and labels
            plt.xlabel('Cross-validation fold', fontsize=fontsize)
            plt.ylabel('Weighted F1 score', fontsize=fontsize)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, fontsize=fontsize-2)
            plt.xticks([1, 2, 3, 4, 5], fontsize=fontsize-2)
            plt.yticks(fontsize=fontsize-2)

            # Display the plot
            #plt.show()
            plt.savefig(os.path.join(datadir, 'Memotion_cv_test_task' + type + '.pdf'), format="pdf", bbox_inches="tight")
        plt.close()

