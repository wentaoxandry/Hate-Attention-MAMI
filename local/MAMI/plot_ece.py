import os, sys, json, argparse
import numpy as np
import scipy
import numpy as np
from reliability_diagrams import reliability_diagram
import logging
from sklearn import metrics


def check_matrix(matrix, gold, pred):
  """Check matrix dimension."""
  if matrix.size == 1:
    tmp = matrix[0][0]
    matrix = np.zeros((2, 2))
    if (pred[1] == 0):
      if gold[1] == 0:  #true negative
        matrix[0][0] = tmp
      else:  #falsi negativi
        matrix[1][0] = tmp
    else:
      if gold[1] == 0:  #false positive
        matrix[0][1] = tmp
      else:  #true positive
        matrix[1][1] = tmp
  return matrix


def compute_f1(pred_values, gold_values):
  matrix = metrics.confusion_matrix(gold_values, pred_values)
  matrix = check_matrix(matrix, gold_values, pred_values)

  #positive label
  if matrix[0][0] == 0:
    pos_precision = 0.0
    pos_recall = 0.0
  else:
    pos_precision = matrix[0][0] / (matrix[0][0] + matrix[0][1])
    pos_recall = matrix[0][0] / (matrix[0][0] + matrix[1][0])

  if (pos_precision + pos_recall) != 0:
    pos_F1 = 2 * (pos_precision * pos_recall) / (pos_precision + pos_recall)
  else:
    pos_F1 = 0.0

  #negative label
  neg_matrix = [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

  if neg_matrix[0][0] == 0:
    neg_precision = 0.0
    neg_recall = 0.0
  else:
    neg_precision = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[0][1])
    neg_recall = neg_matrix[0][0] / (neg_matrix[0][0] + neg_matrix[1][0])

  if (neg_precision + neg_recall) != 0:
    neg_F1 = 2 * (neg_precision * neg_recall) / (neg_precision + neg_recall)
  else:
    neg_F1 = 0.0

  f1 = (pos_F1 + neg_F1) / 2
  return f1

'''def compute_scoreA(taskAdict):
  pred = []
  label = []
  confidence = []
  for i in list(taskAdict.keys()):
    pred.append(taskAdict[i]['predict'])
    label.append(taskAdict[i]['label'])
    confidence.append(taskAdict[i]['prob'][taskAdict[i]['predict']])
  return pred, label, confidence

def predict_scoreA(taskAdict, taskBdict):
    pred = []
    label = []
    confidence = []
    for i in list(taskBdict.keys()):
        prob1 = max(taskBdict[i]['prob'])
        prob = [1-prob1, prob1]
        predict = round(prob1)
        pred.append(predict)
        confidence.append(prob[predict])
        label.append(taskAdict[i]['label'])
    return pred, label, confidence
def compute_scoreB(taskBdict, classtype):
  pred = []
  label = []
  confidence = []
  if classtype == 'Shaming':
    idx = 0
  elif classtype == 'Stereotype':
     idx = 1
  elif classtype == 'Objectification':
    idx = 2
  elif classtype == 'Violence':
    idx = 3

  for i in list(taskBdict.keys()):
    pred.append(taskBdict[i]['predict'][idx])
    label.append(taskBdict[i]['label'][idx])
    confidence.append(taskBdict[i]['prob'][int(taskBdict[i]['predict'][idx])])
  return pred, label, confidence
'''
def compute_scoreA(taskAdict):
  pred = []
  label = []
  confidence = []
  '''for i in list(taskAdict.keys()):
    prob1 = taskAdict[i]['prob'][0]
    prob = [1-prob1, prob1]
    pred.append(prob1)
    if isinstance(taskAdict[i]['predict'], list):
      label.append(taskAdict[i]['label'][0])
      if len(taskAdict[i]['label']) == 1:
        confidence.append(prob[int(taskAdict[i]['predict'][0])])
      else:
        confidence.append(prob[int(taskAdict[i]['predict'][1])])
    else:
       label.append(taskAdict[i]['label'])
       confidence.append(prob[int(taskAdict[i]['predict'])])'''
    
  for i in list(taskAdict.keys()):
      prob1 = taskAdict[i]['prob'][0]
      prob = [1-prob1, prob1]
      pred.append(taskAdict[i]['predict'][0])
      label.append(taskAdict[i]['label'][0])

      confidence.append(prob[int(taskAdict[i]['predict'][0])])
  return pred, label, confidence

def compute_scoreA_early(taskAdict):
  pred = []
  label = []
  confidence = []
   
  for i in list(taskAdict.keys()):
      prob = taskAdict[i]['prob']
      pred.append(taskAdict[i]['predict'])
      label.append(taskAdict[i]['label'])

      confidence.append(prob[int(taskAdict[i]['predict'])])
  return pred, label, confidence

def predict_scoreA(taskAdict, taskBdict):
    pred = []
    label = []
    confidence = []
    for i in list(taskBdict.keys()):
        prob1 = max(taskBdict[i]['prob'])
        prob = [1-prob1, prob1]
        predict = round(prob1)
        if predict == 2:
           predict = predict -1
        pred.append(predict)
        confidence.append(prob[predict])
        if isinstance(taskAdict[i]['label'], list):
          label.append(taskAdict[i]['label'][0])
        else:
          label.append(taskAdict[i]['label'])
    return pred, label, confidence
def compute_scoreB(taskBdict, classtype):
  pred = []
  label = []
  confidence = []
  if classtype == 'Shaming':
    idx = 0
  elif classtype == 'Stereotype':
     idx = 1
  elif classtype == 'Objectification':
    idx = 2
  elif classtype == 'Violence':
    idx = 3

  for i in list(taskBdict.keys()):
    prob1 = taskBdict[i]['prob'][idx]
    prob = [1-prob1, prob1]
    pred.append(taskBdict[i]['predict'][idx])
    label.append(taskBdict[i]['label'][idx])

    confidence.append(prob[int(taskBdict[i]['predict'][idx])])
  return pred, label, confidence

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
    parser.add_argument('--modaltype', default='Unimodal', type=str,
                        help='Dir saves the datasource information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    resultsdir = args.resultsdir
    modaltype = args.modaltype
    savedir = os.path.join(resultsdir, 'ECE_plot')
    if not os.path.exists(savedir):
       os.makedirs(savedir)
    if modaltype is 'Multimodal':
      modals = ['EarlymaskedLM_new',
                  'Hate-CLIPper', 'Hate-Attention-config1', 'Hate-Attention-config2', 'chatgpt', 'chatgpt_zero_shot', 'tiny_llava_tuned']
    else:
       modals = ['mami_text', 'mami_image']
    modalsdir = [os.path.join(resultsdir, modaltype, i) for i in modals]
    for modal, modaldir in zip(modals, modalsdir):
      for task in ['taskA', 'taskB']:
            resultsA = get_data(modaldir, 'taskA')
            resultsB = get_data(modaldir, 'taskB')


            keys = list(resultsA.keys())
            if task == 'taskA':
                if 'Early' in modal:
                  pred, label, confidence =  compute_scoreA_early(resultsA)
                else:
                  pred, label, confidence = compute_scoreA(resultsA)
                #selected_id = [i for i in range(len(label)) if label[i]==1]
                #select_label = [label[i] for i in selected_id]
                #select_pred = [pred[i] for i in selected_id]
                #select_confidence = [confidence[i] for i in selected_id]
                reliability_diagram(label, pred, confidence, savedir, modal + '-task A')


            else:
                for classtype in ['Shaming', 'Stereotype', 'Objectification', 'Violence']:
                    pred, label, confidence =  compute_scoreB(resultsB, classtype)
                    ece = reliability_diagram(label, pred, confidence, savedir, modal + '-' + classtype)
                    logging.info(f'Trained {task} {modal} {classtype} ECE is {ece}')
                pred, label, confidence =  predict_scoreA(resultsA, resultsB)
                reliability_diagram(label, pred, confidence, savedir, modal + '-task AB')

    


       




