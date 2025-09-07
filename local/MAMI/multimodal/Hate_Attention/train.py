import os, sys, json
import random
from model import *
from utils import *
import logging
import numpy as np
import score_computer
from tqdm import tqdm
import argparse
from transformers import get_linear_schedule_with_warmup, CLIPProcessor, CLIPImageProcessor
from transformers import CLIPTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
#import csv
SEED=1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def training(config, mconfig, dataset=None):
    if config["task"] == 'taskA':
        criterion = torch.nn.BCELoss()
        odim = 1
    else:
        scoresham = 10000 / 1274
        scorestere = 10000 / 2810
        scoreobj = 10000 / 2202
        scorevio = 10000 / 953
        weight = torch.FloatTensor([scoresham, scorestere, scoreobj, scorevio])
        criterion = torch.nn.BCELoss(weight=weight.to(config["device"]))
        odim = 4
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    logging.basicConfig(filename=os.path.join(config["modeldir"], 'train.log'), level=logging.INFO)
    evalacc_best = 0
    evalloss_best = float('inf')
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0


    model = CLIP_multi(odim=odim, modelname=config["MODELtext"], cachedir=config["cachedir"], mconfig=mconfig)
    
    '''if config['task'] == 'taskB':
        load_state = torch.load(pretraineddir, map_location='cpu').state_dict()
        self_state = model.state_dict()
        loaded_state = {}
        for k, v in list(load_state.items()):
            if k in self_state and v.size() == self_state[k].size():
                try:
                    loaded_state.update({k: v})
                except:
                    print('error')
        self_state.update(loaded_state)
        model.load_state_dict(self_state)'''
    #for param in model.clip.parameters():
    #    param.requires_grad = False

    #cdcm
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(config['device'])
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')


    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=0.0001
                                  )
    train_examples_len = len(dataset.train_dataset)
    logging.info(f'Train set contains {len(dataset.train_dataset)} samples')
    logging.info(f'Dev set contains {len(dataset.val_dataset)} samples')
    logging.info(f'Test set contains {len(dataset.test_dataset)} samples')
    '''scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 1,
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])'''
    feature_extractor = Feature_extractor(config["image_processor"], config["tokenizer"], config["task"])
    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True, drop_last=False,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=feature_extractor)
    data_loader_dev = torch.utils.data.DataLoader(dataset.val_dataset, shuffle=False, drop_last=False,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=feature_extractor)
    data_loader_test = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=False, drop_last=False,
                                                   batch_size=config["batch_size"],
                                                   num_workers=config["NWORKER"],
                                                   collate_fn=feature_extractor)
    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        torch.cuda.empty_cache()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(tqdm(data_loader_train), 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            pixel = data[2].to(config["device"])
            label = data[3].to(config["device"])

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(node_sets, mask, pixel) #, ITC_loss, ITM_loss
            #label = label.squeeze(-1)

            #if config['task'] == 'taskA':
            #    loss = criterion(torch.sigmoid(outputs), label)
            #elif config['task'] == 'taskB':
            #    loss = criterion(torch.sigmoid(outputs), label)
            loss = criterion(torch.sigmoid(outputs), label)
            #loss = criterion(outputs, label, ITC_loss, ITM_loss)
            #print(loss)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            #scheduler.step()

            #print("\r%f" % loss, end='')

            # print statistics
            tr_loss += loss.item()
            nb_tr_steps += 1
            #if config['task'] == 'taskA':
            #    predicted = torch.round(torch.sigmoid(outputs))# torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            #elif config['task'] == 'taskB':
            #    predicted = torch.round(torch.sigmoid(outputs))
            predicted = torch.round(torch.sigmoid(outputs))
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(label.cpu().detach().data.numpy().tolist())
            del loss, outputs, node_sets, mask, label

        # np.sum(np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1), axis=-1) / len(
        # trainlabel)

        # Validation loss
        print('evaluation')
        torch.cuda.empty_cache()
        evallossvec = []
        evalacc = 0
        model.eval()
        evalpred = []
        evallabel = []
        outpre = {}
        total = 0
        for i, data in enumerate(tqdm(data_loader_dev), 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            pixel = data[2].to(config["device"])
            labels = data[3].to(config["device"])
            filename = data[4]
            outputs = model(node_sets, mask, pixel)
            
            #if config['task'] == 'taskA':
            #    dev_loss = criterion(outputs, labels)
            #    probs = torch.softmax(outputs, dim=-1)
            #    predicted = torch.argmax(probs, dim=-1)
            #elif config['task'] == 'taskB':
            #    dev_loss = criterion(torch.sigmoid(outputs), labels)
            #    probs = torch.sigmoid(outputs)
            #    predicted = torch.round(probs)
            ###
            dev_loss = criterion(torch.sigmoid(outputs), labels)
            #print(loss)
            probs = torch.sigmoid(outputs)
            predicted = torch.round(probs)

            evallossvec.append(dev_loss.cpu().data.numpy())
            evalpred.extend(predicted.cpu().detach().tolist())
            evallabel.extend(labels.cpu().detach().data.numpy().tolist())

            total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': labels[i].cpu().detach().tolist()})
                outpre[filename[i]].update({'predict': predicted[i].cpu().detach().tolist()})
                outpre[filename[i]].update(
                    {'prob': probs[i].cpu().detach().data.numpy().tolist()})
        #del dev_loss, outputs, node_sets, mask, labels


        if config['task'] == 'taskA':
            allscore = score_computer.compute_scoreA(outpre)
        elif config['task'] == 'taskB':
            allscore = score_computer.compute_scoreB(outpre)
        evallossmean = np.mean(np.array(evallossvec))
        logging.info(f'Epoch {epoch} evaluation loss is {evallossmean}, evaluation weighted F1 score is {allscore}.')
        # evalacc = evalacc / len(evallabel)

        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,'bestmodel.pkl')
        #torch.save(model, OUTPUT_DIR)
        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(
                currentlr) + '_' + str(
            allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)

        torch.cuda.empty_cache()
        if allscore <= evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = allscore
            continuescore = continuescore + 1
            torch.save(model, OUTPUT_DIR)
        #if evallossmean >= evalloss_best:
        #    stop_counter = stop_counter + 1
        #    print('no improvement')
        #    continuescore = 0
        #else:
        #    print('new score')
        #    evalloss_best = evallossmean
        #    continuescore = continuescore + 1
        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break

    model = torch.load(os.path.join(modeldir,'bestmodel.pkl'), map_location=config["device"])
    testpred = []
    testlabel = []
    model.eval()
    outpre = {}
    total = 0
    for i, data in enumerate(tqdm(data_loader_test), 0):
        with torch.no_grad():
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            pixel = data[2].to(config["device"])
            labels = data[3].to(config["device"])
            filename = data[4]

            outputs = model(node_sets, mask, pixel)
            
            '''if config['task'] == 'taskA':
                probs = torch.softmax(outputs, dim=-1)
                predicted = torch.argmax(probs, dim=-1)
            elif config['task'] == 'taskB':
                probs = torch.sigmoid(outputs)
                predicted = torch.round(probs)'''
            probs = torch.sigmoid(outputs)
            predicted = torch.round(probs)
            prob = probs.cpu().detach().tolist()
            testlabel.extend(labels.cpu().data.numpy().tolist())
            testpred.extend(predicted.cpu().detach().tolist())
            total += labels.size(0)
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': labels[i].cpu().detach().tolist()})
                outpre[filename[i]].update({'predict': predicted[i].cpu().detach().tolist()})
                outpre[filename[i]].update({'prob': prob[i]})

    testpred = torch.LongTensor(testpred)
    testlabel = torch.LongTensor(testlabel)
    if config['task'] == 'taskA':
        allscore = score_computer.compute_scoreA(outpre)
    elif config['task'] == 'taskB':
        allscore = score_computer.compute_scoreB(outpre)
    testacc = float(allscore)
    logging.info(f'Test weighted F1 score is {testacc}.')
    with open(os.path.join(resultsdir, 'besttestf1_' + str(testacc)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)

def add_tag_info(traintag, testtag, traindict, valdict, testdict):
    for dict in [traindict, valdict]:
        for i in list(dict.keys()):
            filename = i.split('_')[0]
            tags = traintag[filename].split('/')
            tags = ' '.join(tags)
            dict[i].update({'tags': tags})
    for i in list(testdict.keys()):
        filename = i.split('_')[0]
        tags = testtag[filename].split('/')
        tags = ' '.join(tags)
        testdict[i].update({'tags': tags})
    return traindict, valdict, testdict



def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./dataset', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='Hate-CLIPperCA-bimodal', type=str, help='which data stream')
    parser.add_argument('--task', default='taskB', type=str, help='which data stream')
    parser.add_argument('--savedir', default='./output', type=str, help='which data stream')
    parser.add_argument('--cashedir', default='./CASHE', type=str, help='which data stream')
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

    modeldir = os.path.join(savedir, 'Multimodal', modal, task, 'model')
    resultsdir = os.path.join(savedir, 'Multimodal', modal, task, 'results')

    for makedir in [modeldir, resultsdir, cashedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    max_num_epochs = 100
    #print(pretrainedtextdir)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    #device = 'cpu'
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

    #tokenizer = CLIPProcessor.from_pretrained(MODELtext, cache_dir=cashedir)
    image_processor = CLIPImageProcessor.from_pretrained(MODELtext, cache_dir=cashedir)
    tokenizer = CLIPTokenizer.from_pretrained(MODELtext, cache_dir=cashedir)

    dataset = CLIPdatasetclass(train_file=traindict,
                                   val_file=valdict,
                                    test_file=testdict,
                                   device=device,
                                   max_len=max_len, 
                                   task=task)
    '''if task == 'taskA':
        lr = 1e-4
    elif task == 'taskB':
        lr = 1e-4
    #lr = 5e-5'''

    mconfig = {"d_model": 1024, #1024,
               "n_block": 2,
               "n_head": 8,
               "task": task
    }
    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "modal": modal,
        "device": device,
        "lr": 2e-05,
        "modeldir": modeldir,
        "resultsdir": resultsdir,
        "task": task,
        "batch_size": 16, 
        "cachedir": cashedir,
        "epochs": max_num_epochs,
        "image_processor": image_processor,
        "tokenizer": tokenizer
    }
    #, "pretraineddir": pretraineddir
    training(config, mconfig, dataset)








