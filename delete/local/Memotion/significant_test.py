import os, sys
import json
import numpy as np
import argparse, logging
import matplotlib.pyplot as plt
from mlxtend.evaluate import mcnemar_table, mcnemar

def threshold(p_value):
    if p_value <= 0.001:
        sig = '***'
    elif p_value > 0.001 and p_value <= 0.01:
        sig = '**'
    elif p_value > 0.01 and p_value <= 0.05:
        sig = '*'
    elif p_value > 0.05:
        sig = 'ns'
    return sig
def significant_test_taskA(refmodeldir, testmodeldir):
    with open(refmodeldir, encoding="utf8") as json_file:
        reffiledict = json.load(json_file)
    with open(testmodeldir, encoding="utf8") as json_file:
        testfiledict = json.load(json_file)
    true_labels = []
    model1_preds = []
    model2_preds = []
    for key in list(testfiledict.keys()):
        true_labels.append(int(reffiledict[key]['label']))
        model1_preds.append(int(reffiledict[key]['predict']))
        model2_preds.append(int(testfiledict[key]['predict']))

    if 'sarcastic' in refmodeldir:
        model1_preds_new = []
        model2_preds_new = []
        for i in model1_preds:
            if i != 0:
                model1_preds_new.append(3)
            else:
                model1_preds_new.append(0)
        for i in model2_preds:
            if i != 0:
                model2_preds_new.append(3)
            else:
                model2_preds_new.append(0)
        model1_preds = model1_preds_new
        model2_preds = model2_preds_new



    table = mcnemar_table(y_target=np.asarray(true_labels),
                          y_model1=np.asarray(model1_preds),
                          y_model2=np.asarray(model2_preds))

    chi2, p_value = mcnemar(table, exact=False)

    sig = threshold(p_value)
    return sig


def significant_test_taskC(refmodeldir, testmodeldir, labeldir, type):
    with open(refmodeldir, encoding="utf8") as json_file:
        reffiledict = json.load(json_file)
    with open(testmodeldir, encoding="utf8") as json_file:
        testfiledict = json.load(json_file)
    with open(labeldir, encoding="utf8") as json_file:
        labelfiledict = json.load(json_file)
    true_labelsC = []
    true_labelsB = []
    model1_predsC = []
    model2_predsC = []
    for key in list(testfiledict.keys()):
        true_labelsB.append(int(labelfiledict[key]['taskB'][type]))
        true_labelsC.append(int(reffiledict[key]['label']))
        model1_predsC.append(int(reffiledict[key]['predict']))
        model2_predsC.append(int(testfiledict[key]['predict']))

    if 'sarcastic' in refmodeldir:
        model1_predsC_new = []
        model2_predsC_new = []

        for i in model1_predsC:
            if i != 0:
                model1_predsC_new.append(3)
            else:
                model1_predsC_new.append(0)
        for i in model2_predsC:
            if i != 0:
                model2_predsC_new.append(3)
            else:
                model2_predsC_new.append(0)
        model1_predsC = model1_predsC_new
        model2_predsC = model2_predsC_new

    model1_predsB = []
    model2_predsB = []
    for i in model1_predsC:
        if i != 0:
            model1_predsB.append(1)
        else:
            model1_predsB.append(0)
    for i in model2_predsC:
        if i != 0:
            model2_predsB.append(1)
        else:
            model2_predsB.append(0)

    if 'sarcastic' in refmodeldir:
        true_labelsC = [ int(i / 3) for i in true_labelsC]
        model2_predsC = [ int(i / 3) for i in model2_predsC] #model2_predsC / 3
        model1_predsC = [ int(i / 3) for i in model1_predsC] #model1_predsC / 3
    else: 
        pass

    tableC = mcnemar_table(y_target=np.asarray(true_labelsC),
                          y_model1=np.asarray(model1_predsC),
                          y_model2=np.asarray(model2_predsC))
    chi2C, p_valueC = mcnemar(tableC, exact=False)

    tableB = mcnemar_table(y_target=np.asarray(true_labelsB),
                           y_model1=np.asarray(model1_predsB),
                           y_model2=np.asarray(model2_predsB))
    chi2B, p_valueB = mcnemar(tableB, exact=False)

    sigC = threshold(p_valueC)
    sigB = threshold(p_valueB)
    if p_valueB == 0:
        sigB = '-'
    if p_valueC == 0:
        sigC = '-'
    return sigB, sigC, [model1_predsB, model2_predsB, true_labelsB], [model1_predsC, model2_predsC, true_labelsC]


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
    labeldir = './dataset/Memotion3/test_en_labels.json'
    logging.basicConfig(filename=os.path.join(resultsdir, 'significant.log'), level=logging.INFO)
    logging.info(f'{modaltype} F1 scores')



    if modaltype == 'Unimodal':
        modals = ['Challenge-image-0']
        #modals = ['Challenge-image-0', 'Challenge-image-1', 'Challenge-text-0', 'Challenge-text-1', 'memotion_image']
        refmodaldir = os.path.join(resultsdir, modaltype, 'memotion_image')
        refname = 'memotion_image'
    else:
        modals = ['ChatGPT', "ChatGPT_zeroshot", 'TinyLLaVA'] #['DSW_text_RMs', 'DSW_image_RMs', 'DSW_all_RMs', 'DSW_no_RMs',
                 # 'Representation_text_RMs', 'Representation_image_RMs', 'Representation_all_RMs', 'Representation_no_RMs',
                  #'Hate-CLIPper', 'Hate_Attention-config2']
        refmodaldir = os.path.join(resultsdir, modaltype, 'Hate_Attention-config1')
        refname = 'Hate_Attention-config1'
    modalsdir = [os.path.join(resultsdir, modaltype, i) for i in modals]


    ## Multimodal compare ##
    for modal, modaldir in zip(modals, modalsdir):
        for task in ['taskA', 'taskC']:
            if task == 'taskA':
                refmodaldatadir = os.path.join(refmodaldir, task, 'results')
                for i in os.listdir(refmodaldatadir):
                    if 'test' in i:
                        filename = i
                reffiledir = os.path.join(refmodaldatadir, filename)
                testmodeldir = os.path.join(modaldir, task, 'results')
                for i in os.listdir(testmodeldir):
                    if 'test' in i:
                        filename = i
                testfiledir = os.path.join(testmodeldir, filename)
                sigsign = significant_test_taskA(reffiledir, testfiledir)
                logging.info(f'{refname} and {modal} {task} McNemar\'s Test significance: {sigsign}')
            else:
                #allB_label, allB_1, allB_2, allC_label, allC_1, allC_2 = [], [], [], [], [], []
                for type in ['humorous', 'motivation', 'offensive', 'sarcastic']: #, 
                    refmodaldatadir = os.path.join(refmodaldir, task, type, 'results')
                    for i in os.listdir(refmodaldatadir):
                        if 'test' in i:
                            filename = i
                    reffiledir = os.path.join(refmodaldatadir, filename)
                    testmodeldir = os.path.join(modaldir, task, type, 'results')
                    for i in os.listdir(testmodeldir):
                        if 'test' in i:
                            filename = i
                    testfiledir = os.path.join(testmodeldir, filename)
                    sigsignB, sigsignC, taskBdata, taskCdata = significant_test_taskC(reffiledir, testfiledir, labeldir,
                                                                                      type)
                    '''allB_1.extend(taskBdata[0])
                    allB_2.extend(taskBdata[1])
                    allB_label.extend(taskBdata[2])
                    allC_1.extend(taskCdata[0])
                    allC_2.extend(taskCdata[1])
                    allC_label.extend(taskCdata[2])'''
                    logging.info(f'{refname} and {modal} taskB {type} McNemar\'s Test significance: {sigsignB}')
                    logging.info(f'{refname} and {modal} {task} {type} McNemar\'s Test significance: {sigsignC}')

                    '''tableC = mcnemar_table(y_target=np.asarray(allC_label),
                                       y_model1=np.asarray(allC_1),
                                       y_model2=np.asarray(allC_2))
                    chi2C, p_valueC = mcnemar(tableC, exact=False)
                    sigsignC = threshold(p_valueC)
                    logging.info(f'{refname} and {modal} {task} all McNemar\'s Test significance: {sigsignC}')

                    tableB = mcnemar_table(y_target=np.asarray(allB_label),
                                       y_model1=np.asarray(allB_1),
                                       y_model2=np.asarray(allB_2))
                    chi2B, p_valueB = mcnemar(tableB, exact=False)
                    sigsignB = threshold(p_valueB)
                    logging.info(f'{refname} and {modal} taskB all McNemar\'s Test significance: {sigsignB}')
'''
    

