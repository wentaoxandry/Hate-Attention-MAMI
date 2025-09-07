import os, sys, json, argparse
import numpy as np
import scipy, logging
import numpy as np

def paired_t_test(model1_predictions, model2_predictions, ground_truth_labels):
    model1_correct = 0
    model1_incorrect = 0
    model2_correct = 0
    model2_incorrect = 0
    for i in range(len(model1_predictions)):
        if model1_predictions[i] == ground_truth_labels[i]:
            model1_correct = model1_correct + 1
        else:
            model1_incorrect = model1_incorrect + 1
        if model2_predictions[i] == ground_truth_labels[i]:
            model2_correct = model2_correct + 1
        else:
            model2_incorrect = model2_incorrect + 1

    # Calculate the differences between model predictions and ground truth
    table = np.array([[model1_correct, model1_incorrect], [model2_correct, model2_incorrect]])
    tableT = np.transpose(table)
    res = scipy.stats.fisher_exact(table, alternative='two-sided') 
    p_value = res.pvalue
    resT = scipy.stats.fisher_exact(tableT, alternative='two-sided')
    p_valueT = resT.pvalue
    if p_value <= 0.001:
        sig = '***'
    elif p_value > 0.001 and p_value <= 0.01:
        sig = '**'
    elif p_value > 0.01 and p_value <= 0.05:
        sig = '*'
    elif p_value > 0.05:
        sig = 'ns'
    
    return res.statistic, res.pvalue, sig

def constructtaskAdict(taskAdict, taskBdict):
    newtaskA = {}
    for i in list(taskBdict.keys()):
        newtaskA.update({i: {}})
        prob = max(taskBdict[i]['prob'])
        predict = round(prob)
        newtaskA[i].update({'predict': predict})
        newtaskA[i].update({'prob': prob})
        newtaskA[i].update({'label': taskAdict[i]['label']})
    return newtaskA

def get_data(resultsdir, task):
    resultsdir = os.path.join(resultsdir, task, 'results')
    filelist = os.listdir(resultsdir)
    file = [x for x in filelist if 'best' in x][0]
    resultsfiledir = os.path.join(resultsdir, file)
    with open(resultsfiledir, encoding="utf8") as json_file:
        results = json.load(json_file)
    return results

def get_args():
    parser = argparse.ArgumentParser()


    # get arguments from outside
    parser.add_argument('--resultsdir', default='./output/MAMI', type=str,
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
    logging.basicConfig(filename=os.path.join(resultsdir, 'significantunimulti.log'), level=logging.INFO)
    logging.info(f'{modaltype} F1 scores')

    modals = ['mami_text', 'DSW_text_RMs', 'DSW_image_RMs', 'DSW_all_RMs', 'DSW_no_RMs',
                  'Representation_text_RMs', 'Representation_image_RMs', 'Representation_all_RMs', 'Representation_no_RMs']
    refmodaldir = os.path.join(resultsdir, 'Unimodal', 'mami_image')
    refname = 'mami_image'
    resultsA1 = get_data(refmodaldir, 'taskA')
    resultsB1 = get_data(refmodaldir, 'taskB')

    modalsdir = []
    for modal in modals:
        if modal == 'mami_text':
            modalsdir.append(os.path.join(resultsdir, 'Unimodal', modal))
        else:
            modalsdir.append(os.path.join(resultsdir, 'Multimodal', modal))

    for modal, modaldir in zip(modals, modalsdir):
      for task in ['taskA', 'taskB']:
        resultsA2 = get_data(modaldir, 'taskA')
        resultsB2 = get_data(modaldir, 'taskB')
        
        keys = list(resultsA1.keys())
        if task == 'taskA':
            prob1 = []
            prob2 = []
            label = []
            
            for key in keys:
                prob1.append(resultsA1[key]['prob'][0])
                prob2.append(resultsA2[key]['prob'][0])
                label.append(resultsA1[key]['label'][0])
            predict1 = np.round(prob1)
            predict2 = np.round(prob2)
            statistics, p_value, sig = paired_t_test(predict1, predict2, np.array(label))
            logging.info(f'{task} {refname}-{modal} T-statistic: {statistics}, p-value: {p_value}, significance: {sig}')
            # *** signifies p <= 0.001, ** indicates 0.001 < p <= 0.01, * corresponds to 0.01 < p <= 0.05, and ns represents results where p > 0.05.
            

        else:
            prob1 = []
            prob2 = []
            label = []
            for key in keys:
                prob1.append(resultsB1[key]['prob'])
                prob2.append(resultsB2[key]['prob'])
                label.append(resultsB1[key]['label'])
            for i, classname in zip(list(range(4)), ['shaming', 'stereotype', 'objectification', 'violence']):
                subprob1 = [x[i] for x in prob1]
                subprob2 = [x[i] for x in prob2]
                sublabel = [x[i] for x in label]
                predict1 = np.round(subprob1)
                predict2 = np.round(subprob2)
                statistics, p_value, sig = paired_t_test(predict1, predict2, np.array(sublabel))
                logging.info(f'{task} {refname}-{modal} class: {classname} T-statistic: {statistics}, p-value: {p_value}, significance: {sig}')
            prob1 = []
            prob2 = []
            label = []
            newresultsA1 = constructtaskAdict(resultsA1, resultsB1)
            newresultsA2 = constructtaskAdict(resultsA2, resultsB2)
            for key in keys:
                prob1.append(newresultsA1[key]['prob'])
                prob2.append(newresultsA2[key]['prob'])
                if isinstance(newresultsA2[key]['label'], list):
                  label.append(newresultsA2[key]['label'][0])
                else:
                    label.append(newresultsA2[key]['label'])
            predict1 = np.round(prob1)
            predict2 = np.round(prob2)
            statistics, p_value, sig = paired_t_test(predict1, predict2, np.array(label))
            logging.info(f'Predict taskA from {task} {refname}-{modal} T-statistic: {statistics}, p-value: {p_value}, significance: {sig}')
            

