import os, sys, json
from model import *
from utils import *
import argparse
import shutil





def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./../../../dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='mami_bi_opt', type=str, help='which data stream')
    parser.add_argument('--task', default='taskA', type=str, help='which data stream')
    parser.add_argument('--savedir', default='./../../../output', type=str, help='which data stream')
    parser.add_argument('--cashedir', default='./../../../CASHE', type=str, help='which data stream')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    savedir = args.savedir
    task = args.task
    modal = args.modal
    cashedir = args.cashedir

    analysedir = './../../../analyse'
    sourcedir = './../../../Sourcedata/MAMI/test'
    if not os.path.exists(analysedir):
        os.makedirs(analysedir)
    resultsdir = os.path.join(savedir, 'Multimodal', modal, task, 'results')
    for i in os.listdir(resultsdir):
        if i.startswith('best'):
            testresultname = i
    resultsdir = os.path.join(resultsdir, testresultname)

    with open(resultsdir, encoding="utf8") as json_file:
        testresults = json.load(json_file)

    wrongclass = []
    for i in list(testresults.keys()):
        if testresults[i]['label'] != testresults[i]['predict']:
            wrongclass.append(i)

    for i in wrongclass:
        filename = i.split('_')[0]
        srcdir = os.path.join(sourcedir, filename)
        desdir = os.path.join(analysedir, filename)
        shutil.copy(srcdir, desdir)
       




