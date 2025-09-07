import os, sys, json
import random
from model import *
from utils_RAY import *
import logging
import numpy as np
import score_computer
from tqdm import tqdm
import argparse
from transformers import get_linear_schedule_with_warmup, CLIPProcessor, CLIPImageProcessor
from transformers import CLIPTokenizer, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ray
from ray import tune, ray_constants
from ray.tune import CLIReporter
os.environ["RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE"] = "1"
from ray.tune.schedulers import ASHAScheduler
#import csv
SEED=1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_best_trained(filename, mode='acc'):
    filedict = {}
    if mode == 'acc':
        for i in filename:
            score = float(i.split('_')[-1].strip('.pkl'))
            filedict.update({i: score})
        Keymax = max(zip(filedict.values(), filedict.keys()))[1]
    else:
        for i in filename:
            score = float(i.split('_')[-3])
            filedict.update({i: score})
        Keymax = min(zip(filedict.values(), filedict.keys()))[1]
    
    return Keymax, filedict[Keymax]
def training(config, dataset=None):

    mconfig = {"d_model": config["d_model"],
               "n_block": config["n_block"],
               "n_head": config["n_head"],
               "task": config["task"],
    }
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

    #logging.basicConfig(filename=os.path.join(config["modeldir"], 'train.log'), level=logging.INFO)



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
    #logging.info(f'The model has {trainable_parameters} trainable parameters')


    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=0.0001
                                  )
    train_examples_len = len(dataset.train_dataset)
    #logging.info(f'Train set contains {len(dataset.train_dataset)} samples')
    #logging.info(f'Dev set contains {len(dataset.val_dataset)} samples')
    #logging.info(f'Test set contains {len(dataset.test_dataset)} samples')
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


        #with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #    path = os.path.join(checkpoint_dir, "checkpoint")
        #    torch.save((model.state_dict(), optimizer.state_dict()), path)

        ray.train.report(dict(loss=evallossmean, accuracy=allscore))

        #tune.report(loss=evallossmean, accuracy=allscore)
        print("Finished Training")



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

    max_num_epochs = 30
    #print(pretrainedtextdir)

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    #device = 'cpu'

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
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


    '''config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "modal": modal,
        "device": device,
        "lr": lr,
        "task": task,
        "batch_size": 32, #64, #16, #512,  # 16,
        "cachedir": cashedir,
        "epochs": max_num_epochs,
        "image_processor": image_processor,
        "tokenizer": tokenizer,
        "d_model": 1024,
        "dropout": 0.1,
        "n_block": 6,
        "n_head": 16,
        "linear_units": 2048,
        "normalize_before": True,
        "concat_after": False,
        "ifmask": True,
        "polling_type": 'mean',
        "task": task
    }'''

    config = {
        "MODELtext": MODELtext,
        "NWORKER": 0,
        "modal": modal,
        "device": device,
        "lr": tune.choice([1e-4, 5e-5, 2e-5, 1e-5]),
        "task": task,
        "batch_size": tune.choice([8, 16, 32]),
        "cachedir": cashedir,
        "epochs": max_num_epochs,
        "image_processor": image_processor,
        "tokenizer": tokenizer,
        "d_model": tune.choice([64, 128, 256, 512, 768]),
        "n_block": tune.choice([1, 2, 3]),
        "n_head": tune.choice([4, 8, 16]),
        "task": task
    }

    #, "pretraineddir": pretraineddir
    #training(config, dataset)




    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    #training(config, dataset=dataset)
    result = tune.run(
        tune.with_parameters(training, dataset=dataset),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=30,
        storage_path=os.path.join("/media/wentao/Wentaodisk/projekt/MAMI_exp/ChatGPT/output", modal + '_RAY', task),
        scheduler=scheduler,
        progress_reporter=reporter)
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))







