import os, sys, json
import random

import torch.nn
from model import *
from utils import *
import collections
import numpy as np
from sklearn.metrics import  f1_score
from tqdm import tqdm
import argparse
import logging
from transformers import CLIPProcessor
from kaldiio import WriteHelper

SEED=1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_best_trained(filename):
    filedict = {}
    for i in filename:
        score = float(i.split('_')[-1].strip('.pkl'))
        filedict.update({i: score})
    Keymax = max(zip(filedict.values(), filedict.keys()))[1]
    return Keymax, filedict[Keymax]
def training(config, dataset=None):
    saveclsdir = config["saveclsdir"]
    logging.basicConfig(filename=os.path.join(saveclsdir, 'dataset.log'), level=logging.INFO)

    model = CLIPmodel(modelname=config["MODELtext"], cachedir=config["cachedir"])
   
    for param in model.cliptext.parameters():
        param.requires_grad = False

    for param in model.clipimage.parameters():
        param.requires_grad = False


    model.to(config['device'])

    logging.info(f'Train set contains {len(dataset.train_dataset)} samples')
    logging.info(f'Dev set contains {len(dataset.val_dataset)} samples')
    logging.info(f'Test set contains {len(dataset.test_dataset)} samples')


    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True, drop_last=False,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=pad_clip_custom_sequence)
    data_loader_dev = torch.utils.data.DataLoader(dataset.val_dataset, shuffle=True, drop_last=False,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=pad_clip_custom_sequence)

    data_loader_test = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True, drop_last=False,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=pad_clip_custom_sequence)

    arksavetextdir = 'ark,scp:' + os.path.join(saveclsdir, 'text.ark') + ',' + os.path.join(saveclsdir,
                                           'text.scp')
    arksaveimagedir = 'ark,scp:' + os.path.join(saveclsdir, 'image.ark') + ',' + os.path.join(saveclsdir,
                                           'image.scp')
    with WriteHelper(arksavetextdir, compression_method=2) as textwriter:
        with WriteHelper(arksaveimagedir, compression_method=2) as imagewriter:
            model.eval()
            for dataloader in [data_loader_dev, data_loader_test, data_loader_train]:
                for i, data in enumerate(tqdm(dataloader), 0):
                    feats = data[:-2]
                    feats = [i.to(config["device"]) for i in feats]
                    filename = data[-2]
                    uselist = data[-1]
                    text_feature, image_feature = model(feats[0], feats[1], feats[2])
                    text_feature = text_feature.cpu().numpy()
                    image_feature = image_feature.cpu().numpy()
                    for i in range(len(filename)):
                        if uselist[i] == 1:
                            textwriter(filename[i], np.expand_dims(text_feature[i], axis=0))
                            imagewriter(filename[i], np.expand_dims(image_feature[i], axis=0))
                        else:
                            pass


def txt2dict(textdir):
    with open(textdir, "r") as text_file:
	    lines = text_file.readlines()
    newdict = {}
    for i in lines:
        i = i.split(' ')
        newdict.update({i[0]: i[1].strip('\n')})
    return newdict

def get_data(datadict, textdict, imagedict, datakey):
    newdatadict = {}
    for i in datakey:
        newdatadict.update({i: datadict[i]})
        newdatadict[i].update({'textfeat': textdict[i]})
        newdatadict[i].update({'imagefeat': imagedict[i]})
    return newdatadict

def merg_info(traindict, valdict, testdict, textfeatdir, imagefeatdir):
    textdict = txt2dict(textfeatdir)
    imagedict = txt2dict(imagefeatdir)

    trainkey = list(set(list(traindict.keys())) & set(list(textdict.keys())))
    valkey = list(set(list(valdict.keys())) & set(list(textdict.keys())))
    testkey = list(set(list(testdict.keys())) & set(list(textdict.keys())))

    traindict = get_data(traindict, textdict, imagedict, trainkey)
    valdict = get_data(valdict, textdict, imagedict, valkey)
    testdict = get_data(testdict, textdict, imagedict, testkey)
    return traindict, valdict, testdict


def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./dataset/Fakeddit', type=str, help='Dir saves the datasource information')
    parser.add_argument('--cashedir', default='./CASHE', type=str, help='which data stream')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    cashedir = args.cashedir

    saveclsdir = os.path.join(datadir, 'emb_cls')

    for makedir in [cashedir, saveclsdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(os.path.join(datadir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    with open(os.path.join(datadir, "val.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)

    '''import random
    traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 100))}
    testdict = {k: testdict[k] for k in list(random.sample(list(testdict.keys()), 100))}
    valdict = {k: valdict[k] for k in list(random.sample(list(valdict.keys()), 100))}'''

    MODELtext = "openai/clip-vit-large-patch14"
    max_len = 77

    tokenizer = CLIPProcessor.from_pretrained(MODELtext, cache_dir=cashedir)

    dataset = CLIPdatasetclass(train_file=traindict,
                                   val_file=valdict,
                                   test_file=testdict,
                                   tokenizer=tokenizer,
                                   max_len=max_len)

    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "device": device,
        "batch_size": 64,  # 16,
        "cachedir": cashedir,
        "saveclsdir": saveclsdir
    }
    training(config, dataset)

    textfeatdir = os.path.join(saveclsdir, 'text.scp')
    imagefeatdir = os.path.join(saveclsdir, 'image.scp')
    traindict, valdict, testdict = merg_info(traindict, valdict, testdict, textfeatdir, imagefeatdir)
    for dsetdict, dset in zip([valdict, testdict, traindict], ['val', 'test', 'train']):
        with open(os.path.join(datadir, dset + '_feat.json'), "w", encoding="utf-8") as f:
            json.dump(dsetdict, f, ensure_ascii=False, indent=4)









