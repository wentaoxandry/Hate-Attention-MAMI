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

SEED=1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def training(config, dataset=None):
    logging.basicConfig(filename=os.path.join(config["modeldir"], 'train.log'), level=logging.INFO)
    criterion = torch.nn.CrossEntropyLoss()
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0

    if 'text' in config['modal']:
        model = CLIPtext(itdim=768, odim=6, modelname=config["MODELtext"], cachedir=config["cachedir"], hiddendim=768)
    else:
        model = CLIPimage(iidim=1024, odim=6, modelname=config["MODELtext"], cachedir=config["cachedir"], hiddendim=768)
    '''if 'text' in config['modal']:
        for param in model.cliptext.parameters():
            param.requires_grad = False
    else:
        for param in model.clipimage.parameters():
            param.requires_grad = False'''

    #cdcm
    #if torch.cuda.device_count() > 1:
    #    logging.info(f'Let\'s use {torch.cuda.device_count()} GPUs!')
    #    model = torch.nn.DataParallel(model)
    model.to(config['device'])
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')


    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=0.0001
                                  )
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

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        tr_loss = 0
        nb_tr_steps = 0
        model.train()
        pred = []
        labels = []
        for i, data in enumerate(tqdm(data_loader_train), 0):
            label = data[2].to(config["device"])
            optimizer.zero_grad()
            if 'text' in config['modal']:
                textfeat = data[0].to(config["device"])
                outputs, _ = model(textfeat)
            else:
                imagefeat = data[1].to(config["device"])
                outputs, _ = model(imagefeat)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            nb_tr_steps += 1
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            pred.extend(predicted.cpu().tolist())
            labels.extend(label.cpu().tolist())
        train_f1_s = f1_score(np.asarray(labels), np.asarray(pred), average='weighted')
        train_loss = float(tr_loss / nb_tr_steps)
        logging.info(f'Epoch {epoch} training loss is {train_loss}, training weighted F1 score is {train_f1_s}.')

        # Validation loss
        print('evaluation')
        torch.cuda.empty_cache()
        evallossvec = []
        correct = 0
        model.eval()
        evalpred = []
        evallabel = []
        outpre = {}
        total = 0
        for i, data in enumerate(tqdm(data_loader_dev), 0):
            label = data[2].to(config["device"])
            filename = data[3]
            if 'text' in config['modal']:
                textfeat = data[0].to(config["device"])
                outputs, _ = model(textfeat)
            else:
                imagefeat = data[1].to(config["device"])
                outputs, _ = model(imagefeat)
            
            dev_loss = criterion(outputs, label)
            probs = torch.softmax(outputs, dim=-1)
            predicted = torch.argmax(probs, dim=-1)

            evallossvec.append(dev_loss.cpu().data.numpy())
            evalpred.extend(predicted.cpu().tolist())
            evallabel.extend(label.cpu().tolist())
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': label[i].cpu().detach().tolist()})
                outpre[filename[i]].update({'predict': predicted[i].cpu().detach().tolist()})
                outpre[filename[i]].update(
                    {'prob': probs[i].cpu().detach().data.numpy().tolist()})
        eval_f1_s = f1_score(np.asarray(evallabel), np.asarray(evalpred), average='weighted')
        eval_loss = np.mean(np.array(evallossvec))
        logging.info(f'Epoch {epoch} evaluation loss is {eval_loss}, evaluation weighted F1 score is {eval_f1_s}.')

        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,'bestmodel.pkl')

        with open(os.path.join(resultsdir, str(epoch) + '_' + str(eval_loss) + '_' + str(
                currentlr) + '_' + str(
            eval_f1_s)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)

        torch.cuda.empty_cache()
        if eval_f1_s <= evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = eval_f1_s
            continuescore = continuescore + 1
            torch.save(model.state_dict(), OUTPUT_DIR)

        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break

    model.load_state_dict(torch.load(os.path.join(modeldir,'bestmodel.pkl'), map_location=config["device"]))
    testpred = []
    testlabel = []
    model.eval()
    outpre = {}
    for i, data in enumerate(tqdm(data_loader_test), 0):
        with torch.no_grad():
            label = data[2].to(config["device"])
            filename = data[3]
            if 'text' in config['modal']:
                textfeat = data[0].to(config["device"])
                outputs, _ = model(textfeat)
            else:
                imagefeat = data[1].to(config["device"])
                outputs, _ = model(imagefeat)


            probs = torch.softmax(outputs, dim=-1)
            predicted = torch.argmax(probs, dim=-1)
            prob = probs.cpu().detach().tolist()
            testlabel.extend(label.cpu().tolist())
            testpred.extend(predicted.cpu().tolist())
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': label[i].cpu().detach().tolist()})
                outpre[filename[i]].update({'predict': predicted[i].cpu().detach().tolist()})
                outpre[filename[i]].update({'prob': prob[i]})
    test_f1_s = f1_score(np.asarray(testlabel), np.asarray(testpred), average='weighted')
    logging.info(f'Test weighted F1 score is {test_f1_s}.')
    with open(os.path.join(resultsdir,
                                   'test_weighted_f1_score is ' + str(test_f1_s)[:6] + ".json"), 'w',
                      encoding='utf-8') as f:
                json.dump(outpre, f, ensure_ascii=False, indent=4)
    '''for dset, dataloader in zip(['dev', 'train', 'test'], [data_loader_dev, data_loader_train, data_loader_test]):
        model.eval()
        outpre = {}
        for i, data in enumerate(tqdm(dataloader), 0):
            with torch.no_grad():
                if 'text' in config['modal']:
                    inputs_values = data[0].to(config["device"])
                    attention_mask = data[1].to(config["device"])
                    label = data[2].to(config["device"])
                    filename = data[3]
                    outputs = model(inputs_values, attention_mask)
                else:
                    pixel = data[0].to(config["device"])
                    label = data[1].to(config["device"])
                    filename = data[2]
                    outputs = model(pixel)

                for i in range(len(filename)):
                    outpre.update({filename[i]: {}})
                    outpre[filename[i]].update({'logit': outputs[i].cpu().detach().tolist()})
                    outpre[filename[i]].update({'label': label[i].cpu().detach().tolist()})
        with open(os.path.join(config["savefeaturesdir"], dset + 'prob.json'), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)'''

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./dataset/Fakeddit', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='fakeddit_image', type=str, help='which data stream')
    parser.add_argument('--savedir', default='./output/Fakeddit', type=str, help='which data stream')
    parser.add_argument('--cashedir', default='./CASHE', type=str, help='which data stream')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    savedir = args.savedir
    modal = args.modal
    cashedir = args.cashedir

    modeldir = os.path.join(savedir, 'Unimodal', modal, 'model')
    resultsdir = os.path.join(savedir, 'Unimodal', modal, 'results')

    #saveproburesdir = os.path.join(datadir, 'data', modal.split('_')[1])
    #savefeaturesdir = os.path.join(datadir, 'representations', modal.split('_')[1])

    for makedir in [modeldir, resultsdir, cashedir]:#, savefeaturesdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    max_num_epochs = 100
    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(os.path.join(datadir, "train_feat.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "test_feat.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    with open(os.path.join(datadir, "val_feat.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)

    '''import random
    traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 4096))}
    testdict = {k: testdict[k] for k in list(random.sample(list(testdict.keys()), 4096))}
    valdict = {k: valdict[k] for k in list(random.sample(list(valdict.keys()), 4096))}'''

    MODELtext = "openai/clip-vit-large-patch14"
    max_len = 77

    dataset = CLIPdatasetclass(train_file=traindict,
                                   val_file=valdict,
                                   test_file=testdict,
                                   modal=modal,
                                   max_len=max_len)

    lr = 1e-4
    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "modal": modal,
        "device": device,
        "lr": lr,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "batch_size": 64,  # 16,
        "cachedir": cashedir,
        "epochs": max_num_epochs
    }
    #"saveproburesdir": saveproburesdir,
    #"savefeaturesdir": savefeaturesdir,
    training(config, dataset)








