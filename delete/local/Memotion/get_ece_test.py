import os, sys, json, argparse
import numpy as np
import scipy
import numpy as np
from reliability_diagrams import get_ece_score
import logging
from sklearn import metrics


def get_taskA_score(modeldir):
    datadir = os.path.join(modeldir, "taskA", "results")
    for i in os.listdir(datadir):
        if 'test' in i:
            targetfile = i
    with open(os.path.join(datadir, targetfile), encoding="utf8") as json_file:
        resultsdict = json.load(json_file)
    pred = []
    label = []
    confidence = []
    for key in list(resultsdict.keys()):
        label.append(int(resultsdict[key]['label']))
        pred.append(int(resultsdict[key]['predict']))
        confidence.append(resultsdict[key]['prob'][int(resultsdict[key]['predict'])])
    return pred, label, confidence

def get_taskBC_score(modeldir, cate, testdict):
    predB = []
    labelB = []
    confidenceB = []
    predC = []
    labelC = []
    confidenceC = []
        
    datadir = os.path.join(modeldir, "taskC", cate, "results")
    for i in os.listdir(datadir):
        if 'test' in i:
            targetfile = i
    with open(os.path.join(datadir, targetfile), encoding="utf8") as json_file:
        resultsdict = json.load(json_file)

    for key in list(resultsdict.keys()):
            taskCpred = int(resultsdict[key]['predict'])
            taskClab = int(resultsdict[key]['label'])
            taskCconf = resultsdict[key]['prob'][taskCpred]
            if cate == 'sarcastic':
                taskCconf = resultsdict[key]['prob']
                if taskCpred != 0:
                    prob1 = sum(taskCconf[1:])
                    taskCconf = prob1
                    taskCpred = 3
                else:
                    prob1 = taskCconf[0]
                    taskCconf = prob1
                    taskCpred = 0

            if taskCpred != 0:
                taskBpred = 1
                prob1 = max(resultsdict[key]['prob'][1:])
                prob = [1-prob1, prob1]
                taskBconf = prob[taskBpred]
            else:
                taskBpred = taskCpred
                taskBconf = taskCconf
            taskBlab = int(testdict[key]['taskB'][cate])
            predB.append(taskBpred)
            labelB.append(taskBlab)
            confidenceB.append(taskBconf)

            predC.append(taskCpred)
            labelC.append(taskClab)
            confidenceC.append(taskCconf)
        
    return predB, labelB, confidenceB, predC, labelC, confidenceC

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
    logging.basicConfig(filename=os.path.join(resultsdir, 'ECE.log'), level=logging.INFO)
    logging.info(f'Unimodal ECE scores')

    if modaltype == 'Unimodal':
        modals = ['memotion_image', 'memotion_text']
    else:
        modals = ['ensemble_1'] #'DSW_text_RMs', 'DSW_image_RMs', 'DSW_all_RMs', 'DSW_no_RMs',
                  #'Representation_text_RMs', 'Representation_image_RMs', 'Representation_all_RMs', 'Representation_no_RMs',
                  #'Hate-CLIPper', 'Hate_Attention-config1', 'Hate_Attention-config2', , 'ChatGPT', "ChatGPT_zeroshot", 'TinyLLaVA'
    modalsdir = [os.path.join(resultsdir, modaltype, i) for i in modals]
    with open(os.path.join('./dataset/Memotion3/test_en_labels.json'), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    results = {}
    for modal, modaldir in zip(modals, modalsdir):
        results.update({modal: {}})
        for task in ['taskA', 'taskC']:
            if task == "taskA":
                pred, label, confidence = get_taskA_score(modaldir)
                ece = get_ece_score(label, pred, confidence, modal)
                results[modal].update({task: ece})
                #logging.info(f'{modal} {task} ece score is {ece}')
            else:
                results[modal].update({'taskB': {}})
                results[modal].update({'taskC': {}})
                for cate in os.listdir(os.path.join(modaldir, 'taskC')):
                    predB, labelB, confidenceB, predC, labelC, confidenceC = get_taskBC_score(modaldir, cate, testdict)
                    confidenceB = np.asarray(confidenceB)
                    confidenceC = np.asarray(confidenceC)
                        
                        
                    eceB = get_ece_score(labelB, predB, confidenceB, modal)
                    eceC = get_ece_score(labelC, predC, confidenceC, modal)
                    results[modal]['taskB'].update({cate: eceB})
                    results[modal]['taskC'].update({cate: eceC})
                    #logging.info(f'{modal} TaskB {cate} F1 score is {eceB}')
                    #logging.info(f'{modal} TaskC {cate} F1 score is {eceC}')

    for modal in list(results.keys()):
        logging.info(f'{modal} TaskA ece score is {results[modal]["taskA"] * 100}')
        logging.info(f'{modal} TaskB humorous ece score is {results[modal]["taskB"]["humorous"] * 100}')
        logging.info(f'{modal} TaskB motivation ece score is {results[modal]["taskB"]["motivation"] * 100}')
        logging.info(f'{modal} TaskB offensive ece score is {results[modal]["taskB"]["offensive"] * 100}')
        logging.info(f'{modal} TaskB sarcastic ece score is {results[modal]["taskB"]["sarcastic"] * 100}')
        logging.info(f'{modal} TaskC humorous ece score is {results[modal]["taskC"]["humorous"] * 100}')
        logging.info(f'{modal} TaskC motivation ece score is {results[modal]["taskC"]["motivation"] * 100}')
        logging.info(f'{modal} TaskC offensive ece score is {results[modal]["taskC"]["offensive"] * 100}')
        logging.info(f'{modal} TaskC sarcastic ece score is {results[modal]["taskC"]["sarcastic"] * 100}')
    a = 1


       




