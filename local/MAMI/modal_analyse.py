import os, sys, json, argparse
import numpy as np
import scipy
import numpy as np
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
  guess = []    #predict
  gold = []     #ground-truth
  for i in list(taskAdict.keys()):
    guess.append(taskAdict[i]['predict'])
    gold.append(taskAdict[i]['label'])
  score = compute_f1(guess, gold)
  #print('Task A score is ' + str(round(score, 5)))
  return score

def predict_scoreA(taskAdict, taskBdict):
    newtaskA = {}
    for i in list(taskBdict.keys()):
        newtaskA.update({i: {}})
        prob = max(taskBdict[i]['prob'])
        predict = round(prob)
        newtaskA[i].update({'predict': predict})
        newtaskA[i].update({'prob': prob})
        newtaskA[i].update({'label': taskAdict[i]['label']})
    return compute_scoreA(newtaskA)
def compute_scoreB(taskBdict):
  results = []
  total_occurences = 0
  catescores = []
  for index in range(4):
    guess = []    #predict
    gold = []     #ground-truth
    for i in list(taskBdict.keys()):
        guess.append(taskBdict[i]['predict'][index])
        gold.append(taskBdict[i]['label'][index])
    f1_score = compute_f1(guess, gold)
    catescores.append(f1_score)
    weight = gold.count(True)
    total_occurences += weight
    results.append(f1_score * weight)
  score = sum(results) / total_occurences
  return score, catescores

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
    logging.basicConfig(filename=os.path.join(resultsdir, 'F1 score.log'), level=logging.INFO)
    logging.info(f'Unimodal F1 scores')

    if modaltype == 'Unimodal':
        modals = ['BERTC', 'GCAN', 'mami_text', 'mami_image', 'ViT']
    else:
        modals = ['EarlymaskedLM_new', 'DSW_text_RMs', 'DSW_image_RMs', 'DSW_all_RMs', 'DSW_no_RMs',
                  'Representation_text_RMs', 'Representation_image_RMs', 'Representation_all_RMs', 'Representation_no_RMs',
                  'Hate-CLIPper', 'Hate-Attention-config1', 'Hate-Attention-config2', 'chatgpt', 'chatgpt_zero_shot', 'tiny_llava_tuned']
    modalsdir = [os.path.join(resultsdir, modaltype, i) for i in modals]

    for modal, modaldir in zip(modals, modalsdir):
        for task in ['taskA', 'taskB']:
            resultsA = get_data(modaldir, 'taskA')
            resultsB = get_data(modaldir, 'taskB')

            pred = []
            label = []
            keys = list(resultsA.keys())
            if task == 'taskA':
                allscore = compute_scoreA(resultsA)
                logging.info(f'Trained {task} {modal} F1 score is {allscore}')

            else:
                taskBallscore, taskBcatescore = compute_scoreB(resultsB)
                taskAallscore = predict_scoreA(resultsA, resultsB)
                logging.info(f'Trained {task} {modal} F1 score is {taskBallscore}')
                logging.info(f'Trained {task} {modal} F1 score for shaming is {taskBcatescore[0]}')
                logging.info(f'Trained {task} {modal} F1 score for stereotype is {taskBcatescore[1]}')
                logging.info(f'Trained {task} {modal} F1 score for objectification is {taskBcatescore[2]}')
                logging.info(f'Trained {task} {modal} F1 score for violence is {taskBcatescore[3]}')
                logging.info(f'Predict taskA from {task} {modal} F1 score is {taskAallscore}')






