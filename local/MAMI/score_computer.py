import numpy as np
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

def compute_scoreB(taskBdict):
  results = []
  total_occurences = 0
  for index in range(4):
    guess = []    #predict
    gold = []     #ground-truth
    for i in list(taskBdict.keys()):
        guess.append(taskBdict[i]['predict'][index])
        gold.append(taskBdict[i]['label'][index])
    f1_score = compute_f1(guess, gold)
    weight = gold.count(True)
    total_occurences += weight
    results.append(f1_score * weight)
  score = sum(results) / total_occurences
  return score

