import os, json, argparse
import numpy as np
import numpy as np
from reliability_diagrams import get_ece_score
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

def compute_scoreA(taskAdict):
  pred = []
  label = []
  confidence = []    
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
        modals = ['mami_image', 'mami_text']
    else:
        modals = ['Hate-CLIPper', 'Hate-Attention', 'Tiny-LLaVA', 'ChatGPT/zero_shot', 'ChatGPT/few_shot']
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
                ece = get_ece_score(label, pred, confidence, modal)
                logging.info(f'Trained {task} {modal} ECE is {ece}')
            else:
                for classtype in ['Shaming', 'Stereotype', 'Objectification', 'Violence']:
                    pred, label, confidence =  compute_scoreB(resultsB, classtype)
                    ece = get_ece_score(label, pred, confidence, modal)
                    logging.info(f'Trained {task} {modal} {classtype} ECE is {ece}')
                pred, label, confidence =  predict_scoreA(resultsA, resultsB)
                ece = get_ece_score(label, pred, confidence, modal)
                logging.info(f'Predict taskA from {task} {modal}  ECE is {ece}')

    


       




