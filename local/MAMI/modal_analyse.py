import os, json, argparse
import logging
from score_computer import *

def predict_scoreA(taskAdict, taskBdict):
    """
    Predict task A score using task B probabilities.

    For each sample, the maximum probability among taskB classes is used as a
    binary posterior (rounded) to construct a taskA-style prediction dictionary,
    which is then scored with `compute_scoreA`.

    Parameters
    ----------
    taskAdict : dict
        Dictionary containing task A ground-truth labels per sample.
    taskBdict : dict
        Dictionary containing task B probabilities per sample.
    Returns
    -------
    float
        Task A F1 score computed by `compute_scoreA`.
    """
    newtaskA = {}
    for i in list(taskBdict.keys()):
        newtaskA.update({i: {}})
        prob = max(taskBdict[i]['prob'])
        predict = round(prob)
        newtaskA[i].update({'predict': predict})
        newtaskA[i].update({'prob': prob})
        newtaskA[i].update({'label': taskAdict[i]['label']})
    return compute_scoreA(newtaskA)


def get_data(resultsdir, task):
    """
    Load the best results JSON for a given task from a model's results directory.

    Parameters
    ----------
    resultsdir : str
        Base directory for a specific model under which task results are stored.
    task : str
        Task name, e.g., "taskA" or "taskB".

    Returns
    -------
    dict
        Parsed JSON dictionary of predictions/probabilities/labels for the task.
    """
    resultsdir = os.path.join(resultsdir, task, 'results')
    filelist = os.listdir(resultsdir)
    file = [x for x in filelist if 'best' in x][0]
    resultsfiledir = os.path.join(resultsdir, file)
    with open(resultsfiledir, encoding="utf8") as json_file:
        results = json.load(json_file)
    return results

def get_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with fields:
        - resultsdir : str
            Root output directory containing model results.
        - modaltype : str
            Either "Unimodal" or "Multimodal".
    """
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
        modals = ['mami_text', 'mami_image']
    else:
        modals = ['Hate-CLIPper', 'Hate-Attention', 'Tiny-LLaVA', 'ChatGPT/zero_shot', 'ChatGPT/few_shot']
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






