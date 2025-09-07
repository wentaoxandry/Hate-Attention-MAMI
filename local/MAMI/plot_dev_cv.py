import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import json
from sklearn import metrics
import seaborn as sns
def find_best_trained(filename):
    filedict = {}
    for i in filename:
        score = float(i.split('_')[-1].strip('.json'))
        filedict.update({i: score})
    Keymax = max(zip(filedict.values(), filedict.keys()))[1]
    return Keymax
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


def predict_scoreA(labeldict, taskBdict):
    newtaskA = {}
    for i in list(taskBdict.keys()):
        newtaskA.update({i: {}})
        prob = max(taskBdict[i]['prob'])
        predict = round(prob)
        newtaskA[i].update({'predict': predict})
        newtaskA[i].update({'prob': prob})
        newtaskA[i].update({'label': int(labeldict[i]['taskA'])})
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
  return score#, catescores

def get_data(datadir, modal):
    modals = []
    task = []
    score = []
    for i in range(10):
        subresultsAdir = os.path.join(datadir, 'Multimodal', modal, str(i), 'train_10', 'taskA')
        subresultsBdir = os.path.join(datadir, 'Multimodal', modal, str(i), 'train_10', 'taskB')
    
        resultsAdir = os.path.join(subresultsAdir, 'results')
        fileAlist = os.listdir(resultsAdir)
        fileA = [x for x in fileAlist if 'best' in x][0]
        resultsfileAdir = os.path.join(resultsAdir, fileA)
        with open(resultsfileAdir, encoding="utf8") as json_file:
            resultsA = json.load(json_file)

        resultsBdir = os.path.join(subresultsBdir, 'results')
        fileBlist = os.listdir(resultsBdir)
        fileB = [x for x in fileBlist if 'best' in x][0]
        resultsfileBdir = os.path.join(resultsBdir, fileB)
        with open(resultsfileBdir, encoding="utf8") as json_file:
            resultsB = json.load(json_file)
        modals.append(modal)
        task.append('Task A')
        score.append(predict_scoreA(resultsA, resultsB))
        modals.append(modal)
        task.append('Task B')
        score.append(compute_scoreB(resultsB))

    return modals, task, score
def get_data_norm(datadir, modal, labels):
    scoreA = []
    scoreB = []
    for i in range(10):
        subresultsBdir = os.path.join(datadir, 'Multimodal', modal, 'taskB', str(i))

        resultsBdir = os.path.join(subresultsBdir, 'results')
        fileBlist = os.listdir(resultsBdir)
        fileBs = [x for x in fileBlist if 'best' not in x]
        fileB = find_best_trained(fileBs)
        resultsfileBdir = os.path.join(resultsBdir, fileB)
        with open(resultsfileBdir, encoding="utf8") as json_file:
            resultsB = json.load(json_file)
        scoreA.append(predict_scoreA(labels, resultsB))

        scoreB.append(compute_scoreB(resultsB))

    return scoreA, scoreB
def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./../../output/MAMI', type=str, help='Dir saves the datasource information')
    parser.add_argument('--sourcedir', default='./../../dataset/MAMI', type=str, help='Dir saves the datasource information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]
    task = 'taskB'
    datadir = args.datadir
    sourcedir = args.sourcedir
    fontsize = 15

    with open(os.path.join(sourcedir, 'train.json'), encoding="utf8") as json_file:
      labels = json.load(json_file)

    with open(os.path.join(sourcedir, 'val.json'), encoding="utf8") as json_file:
      labelsdev = json.load(json_file)
    labels.update(labelsdev)
    models = ['Hate-CLIPper-cv', 'Hate-Attention-cv-config1', 'Hate-Attention-cv-config2']

    allscores = {}
    for model in models:
        allscores.update({model: {}})
        scoreA, scoreB = get_data_norm(datadir, model, labels)
        allscores[model].update({'taskA': scoreA})
        allscores[model].update({'taskB': scoreB})

    for model in models:
      score = allscores[model][task]
      cvs = list(range(10))
      cvs = [k + 1 for k in cvs]
      if model == 'Hate-CLIPper-cv':
         model = 'Hate-CLIPper'
      elif model == 'Hate-Attention-cv-config1':
         model = 'Hate-Attention-config1'
      elif model == 'Hate-Attention-cv-config2':
         model = 'Hate-Attention-config2'
      plt.plot(cvs, score, label=model)
         
    # Adding title and labels
    plt.xlabel('Cross-validation fold', fontsize=fontsize)
    plt.ylabel('Weighted F1 score', fontsize=fontsize)

    # Adding legend
    plt.legend(fontsize=fontsize, bbox_to_anchor=(0.5, 1.15), loc='center', ncol=1)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Display the plot
    #plt.show()
    plt.savefig(os.path.join(datadir, 'cv_eval_' + task + '.pdf'), bbox_inches='tight')
