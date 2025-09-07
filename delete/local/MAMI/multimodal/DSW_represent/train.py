import os, sys, json
import random
from model import *
from utils import *
import score_computer
import numpy as np
from tqdm import tqdm
import argparse
from transformers import get_linear_schedule_with_warmup, CLIPProcessor
from transformers import CLIPTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau
#import csv
SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def training(config, dataset=None):
    logging.basicConfig(filename=os.path.join(config["modeldir"], 'train.log'), level=logging.INFO)
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
    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0

    if 'DSW' in config['modal']:
        model = DSW(itdim=768, iidim=1024, odim=odim, modelname=config["MODELtext"], cachedir=config["cachedir"], hiddendim=32, RMs=config['RMs'], task=config["task"])
    elif 'Representation' in config['modal']:
        model = Representation(itdim=768, iidim=1024, odim=odim, modelname=config["MODELtext"], cachedir=config["cachedir"], hiddendim=768, RMs=config['RMs'], task=config["task"])

    # load model
    pretrainedimgdir = config["pretrainedimgdir"]
    pretrainedtextdir = config["pretrainedtextdir"]
    mcliptextstatedict = model.textclassifier.state_dict()
    cliptextstatedict = torch.load(pretrainedtextdir, map_location='cpu').state_dict()
    filtered_state_dict = {k: v for k, v in cliptextstatedict.items() if k in mcliptextstatedict}
    model.textclassifier.load_state_dict(filtered_state_dict)

    mclipimagestatedict = model.imageclassifier.state_dict()
    clipimagestatedict = torch.load(pretrainedimgdir, map_location='cpu').state_dict()
    filtered_state_dict = {k: v for k, v in clipimagestatedict.items() if k in mclipimagestatedict}
    model.imageclassifier.load_state_dict(filtered_state_dict)
    
    # Laden der gefilterten state_dict
    #model_state_dict.update(filtered_state_dict)
    #model.load_state_dict(model_state_dict)


    #model.clipimage = torch.load(pretrainedimgdir, map_location='cpu')
    #imagestatedict = torch.load(pretrainedimgdir, map_location='cpu')
    #modelstatedict = model.state_dict()
    #modelstatedict.update(textstatedict)
    #modelstatedict.update(imagestatedict)
    #model.load_state_dict(modelstatedict)
    

    #cdcm
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(config['device'])
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')


    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"]
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

    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True, drop_last=True,
                                                    batch_size=config["batch_size"],
                                                    num_workers=config["NWORKER"],
                                                    collate_fn=pad_clip_custom_sequence)
    data_loader_dev = torch.utils.data.DataLoader(dataset.val_dataset, shuffle=True, drop_last=True,
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
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(tqdm(data_loader_train), 0):
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            pixel = data[2].to(config["device"])
            score = data[3].to(config["device"])
            label = data[4].to(config["device"])

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(node_sets, mask, pixel, score)


            loss = criterion(torch.sigmoid(outputs), label)
            loss.backward()
            optimizer.step()
            #scheduler.step()

            #print("\r%f" % loss, end='')

            # print statistics
            #tr_loss += loss.item()
            #nb_tr_steps += 1
            #if config['task'] == 'taskA':
            #    predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            #elif config['task'] == 'taskB':
            #    predicted = torch.round(torch.sigmoid(outputs))
            #trainpredict.extend(predicted.cpu().detach().tolist())
            #trainlabel.extend(label.cpu().detach().data.numpy().tolist())
            del loss, outputs, node_sets, mask, label
        #trainallscore = f1_score(trainlabel, trainpredict, average='weighted')

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
            score = data[3].to(config["device"])
            labels = data[4].to(config["device"])
            filename = data[5]
            outputs = model(node_sets, mask, pixel, score)
            
 
            dev_loss = criterion(torch.sigmoid(outputs), labels)
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


        #allscore = f1_score(evallabel, evalpred, average='weighted') 
        if config['task'] == 'taskA':
            allscore = score_computer.compute_scoreA(outpre)
        elif config['task'] == 'taskB':
            allscore = score_computer.compute_scoreB(outpre)
        # evalacc = evalacc / len(evallabel)
        evallossmean = np.mean(np.array(evallossvec))
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        logging.info(f'Epoch {epoch} evaluation loss is {evallossmean}, evaluation weighted F1 score is {allscore}.')
        OUTPUT_DIR = os.path.join(modeldir,'bestmodel.pkl')

        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(
                currentlr) + '_' + str(
            allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outpre, f, ensure_ascii=False, indent=4)

        torch.cuda.empty_cache()
        if allscore < evalacc_best:
            stop_counter = stop_counter + 1
            print('no improvement')
            continuescore = 0
        else:
            print('new score')
            evalacc_best = allscore
            continuescore = continuescore + 1
            torch.save(model, OUTPUT_DIR)

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
            score = data[3].to(config["device"])
            labels = data[4].to(config["device"])
            filename = data[5]

            outputs = model(node_sets, mask, pixel, score)
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
    with open(os.path.join(resultsdir,
                           'besttestf1_' + str(testacc)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)


def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./dataset/MAMI', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='DSW', type=str, help='which data stream')
    parser.add_argument('--task', default='taskB', type=str, help='which data stream')
    parser.add_argument('--RMs', default='text_RMs', type=str, help="possible are 'all_RMs', 'text_RMs', 'image_RMs' and 'no_RMs'")
    parser.add_argument('--savedir', default='./output/MAMI', type=str, help='which data stream')
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
    RMs = args.RMs                  
    modal = modal + '_' + RMs
    cashedir = args.cashedir

    modeldir = os.path.join(savedir, 'Multimodal', modal, task, 'model')
    resultsdir = os.path.join(savedir, 'Multimodal', modal, task, 'results')

    pretrainedtextdir = os.path.join(savedir, 'Unimodal', 'mami_text', task, 'model', 'bestmodel.pkl')
    pretrainedimgdir = os.path.join(savedir, 'Unimodal', 'mami_image', task, 'model', 'bestmodel.pkl')

    for makedir in [modeldir, resultsdir, cashedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    max_num_epochs = 70

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    scoresdir = os.path.join(datadir, 'RMs')
    with open(os.path.join(datadir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    with open(os.path.join(datadir, "val.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)

    with open(os.path.join(scoresdir, "text_train.json"), encoding="utf8") as json_file:
        texttrainscoredict = json.load(json_file)
    with open(os.path.join(scoresdir, "text_dev.json"), encoding="utf8") as json_file:
        textdevscoredict = json.load(json_file)
    texttrainscoredict.update(textdevscoredict)
    with open(os.path.join(scoresdir, "text_test.json"), encoding="utf8") as json_file:
        texttestscoredict = json.load(json_file)
    texttrainscoredict.update(texttestscoredict)
    with open(os.path.join(scoresdir, "image_train.json"), encoding="utf8") as json_file:
        imagetrainscoredict = json.load(json_file)
    with open(os.path.join(scoresdir, "image_dev.json"), encoding="utf8") as json_file:
        imagedevscoredict = json.load(json_file)
    imagetrainscoredict.update(imagedevscoredict)
    with open(os.path.join(scoresdir, "image_test.json"), encoding="utf8") as json_file:
        imagetestscoredict = json.load(json_file)
    imagetrainscoredict.update(imagetestscoredict)

    for dset, workdict in zip(['train', 'dev', 'test'], [traindict, valdict, testdict]):
        for key in list(workdict.keys()):
            workdict[key].update({'textscore': texttrainscoredict[key]})
            workdict[key].update({'imagescore': imagetrainscoredict[key]})
        if dset == 'train':
            traindict = workdict
        elif dset == 'dev':
            valdict = workdict
        elif dset == 'test':
            testdict = workdict


    MODELtext = "openai/clip-vit-large-patch14"
    max_len = 77

    tokenizer = CLIPProcessor.from_pretrained(MODELtext, cache_dir=cashedir)

    dataset = CLIPdatasetclass(train_file=traindict,
                                   val_file=valdict,
                                   test_file=testdict,
                                   tokenizer=tokenizer,
                                   device=device,
                                   max_len=max_len, 
                                   task=task)
    if task == 'taskA':
        lr = 1e-4
    elif task == 'taskB':
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
        "batch_size": 64, #512,  # 16,
        "cachedir": cashedir,
        "RMs": RMs,
        "epochs": max_num_epochs,
        "task": task,
        "pretrainedtextdir": pretrainedtextdir,
        "pretrainedimgdir": pretrainedimgdir
    }
    training(config, dataset)








