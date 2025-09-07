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
from transformers import CLIPTokenizer, CLIPImageProcessor

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
    logging.basicConfig(filename=os.path.join(config["modeldir"], 'train.log'), level=logging.INFO)
    criterion = torch.nn.CrossEntropyLoss()
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    if config["classname"] == 'motivation':
        odim = 2
    else:
        odim = 4

    model = CLIP_multi(odim=odim, modelname=config["MODELtext"], cachedir=config["cachedir"])
    for param in model.clip.parameters():
        param.requires_grad = False

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

    feature_extractor = Feature_extractor(config["image_processor"], config["tokenizer"])
    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True, drop_last=False,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=feature_extractor)
    data_loader_dev = torch.utils.data.DataLoader(dataset.val_dataset, shuffle=True, drop_last=False,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=feature_extractor)

    data_loader_test = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True, drop_last=False,
                                                  batch_size=config["batch_size"],
                                                  num_workers=config["NWORKER"],
                                                  collate_fn=feature_extractor)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        tr_loss = 0
        nb_tr_steps = 0
        model.train()
        pred = []
        labels = []
        for i, data in enumerate(tqdm(data_loader_train), 0):
            inputs_values = data[0].to(config["device"])
            attention_mask = data[1].to(config["device"])
            pixel = data[2].to(config["device"])
            labelA = data[3].to(config["device"])
            labelB1 = data[4].to(config["device"])
            labelB2 = data[5].to(config["device"])
            labelB3 = data[6].to(config["device"])
            labelB4 = data[7].to(config["device"])
            filename = data[8]
            if config["classname"] == 'humorous':
                label = labelB1
            elif config["classname"] == 'sarcastic':
                label = labelB2
            elif config["classname"] == 'offensive':
                label = labelB3
            elif config["classname"] == 'motivation':
                label = labelB4
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs_values, attention_mask, pixel)

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
            inputs_values = data[0].to(config["device"])
            attention_mask = data[1].to(config["device"])
            pixel = data[2].to(config["device"])
            labelA = data[3].to(config["device"])
            labelB1 = data[4].to(config["device"])
            labelB2 = data[5].to(config["device"])
            labelB3 = data[6].to(config["device"])
            labelB4 = data[7].to(config["device"])
            filename = data[8]
            if config["classname"] == 'humorous':
                label = labelB1
            elif config["classname"] == 'sarcastic':
                label = labelB2
            elif config["classname"] == 'offensive':
                label = labelB3
            elif config["classname"] == 'motivation':
                label = labelB4

            outputs = model(inputs_values, attention_mask, pixel)
            
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
        if eval_f1_s < evalacc_best:
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
            inputs_values = data[0].to(config["device"])
            attention_mask = data[1].to(config["device"])
            pixel = data[2].to(config["device"])
            labelA = data[3].to(config["device"])
            labelB1 = data[4].to(config["device"])
            labelB2 = data[5].to(config["device"])
            labelB3 = data[6].to(config["device"])
            labelB4 = data[7].to(config["device"])
            filename = data[8]
            if config["classname"] == 'humorous':
                label = labelB1
            elif config["classname"] == 'sarcastic':
                label = labelB2
            elif config["classname"] == 'offensive':
                label = labelB3
            elif config["classname"] == 'motivation':
                label = labelB4

            outputs = model(inputs_values, attention_mask, pixel)

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


def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./../../dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='HATE-CLIPper', type=str, help='which data stream')
    parser.add_argument('--task', default='taskA', type=str, help='which data stream')
    parser.add_argument('--classname', default='humorous', type=str, help='which data stream')
    parser.add_argument('--savedir', default='./../../output', type=str, help='which data stream')
    parser.add_argument('--cashedir', default='./../../CASHE', type=str, help='which data stream')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    savedir = args.savedir
    task = args.task
    classname = args.classname
    modal = args.modal
    cashedir = args.cashedir

    modeldir = os.path.join(savedir, 'Multimodal', modal, task, classname, 'model')
    resultsdir = os.path.join(savedir, 'Multimodal', modal, task, classname, 'results')

    for makedir in [modeldir, resultsdir, cashedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    max_num_epochs = 100
    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    with open(os.path.join(datadir, "train_en.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "test_en_labels.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    with open(os.path.join(datadir, "val_en.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)

    '''import random
    traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 100))}
    testdict = {k: testdict[k] for k in list(random.sample(list(testdict.keys()), 100))}
    valdict = {k: valdict[k] for k in list(random.sample(list(valdict.keys()), 100))}'''

    MODELtext = "openai/clip-vit-large-patch14"
    max_len = 77

    image_processor = CLIPImageProcessor.from_pretrained(MODELtext, cache_dir=cashedir)
    tokenizer = CLIPTokenizer.from_pretrained(MODELtext, cache_dir=cashedir)

    dataset = CLIPdatasetclass(train_file=traindict,
                                   val_file=valdict,
                                   test_file=testdict,
                                   tokenizer=tokenizer,
                                   device=device,
                                   max_len=max_len, 
                                   task=task)

    lr = 1e-4
    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "modal": modal,
        "device": device,
        "lr": lr,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "task": task,
        "batch_size": 64,  # 16,
        "classname": classname,
        "cachedir": cashedir,
        "image_processor": image_processor,
        "tokenizer": tokenizer,
        "epochs": max_num_epochs
    }
    training(config, dataset)








