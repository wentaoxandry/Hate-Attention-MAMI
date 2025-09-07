import os, json
import random
from model import *
from utils import *
import logging
import numpy as np
from ... import score_computer
from tqdm import tqdm
import argparse
from transformers import CLIPImageProcessor, CLIPTokenizer

SEED=1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def run(data_loader, model, config, criterion, train=True, optimizer=None):
    """
    Run one epoch of training or evaluation.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        DataLoader providing batches of training or evaluation data.
    model : torch.nn.Module
        Model to be trained or evaluated.
    config : dict
        Configuration.
    criterion : torch.nn.Module
        Loss function used for training/evaluation.
    train : bool, default=True
        Whether to run in training mode (updates weights) or evaluation mode.
    optimizer : torch.optim.Optimizer
        Optimizer used for parameter updates (only when `train=True`).

    Returns
    -------
    model : torch.nn.Module
        The model after running one epoch (updated if `train=True`).
    lossmean : float
        Mean loss over all batches in the epoch.
    allscore : float
        Computed evaluation score (depends on task type).
    outpre : dict
        Dictionary mapping filenames to predictions, labels, and probabilities.
    """    
    lossvec = []
    outpre = {}
    for i, data in enumerate(tqdm(data_loader), 0):
        node_sets = data[0].to(config["device"])
        mask = data[1].to(config["device"])
        pixel = data[2].to(config["device"])
        label = data[3].to(config["device"])
        filename = data[4]

        # zero the parameter gradients
        if train:
            optimizer.zero_grad()
            
        outputs = model(node_sets, mask, pixel)

        loss = criterion(torch.sigmoid(outputs), label)
        lossvec.append(loss.cpu().data.numpy())

        if train:
            loss.backward()
            optimizer.step()

        probs = torch.sigmoid(outputs)
        predicted = torch.round(probs)

        for i in range(len(filename)):
            outpre.update({filename[i]: {}})
            outpre[filename[i]].update({'label': label[i].cpu().detach().tolist()})
            outpre[filename[i]].update({'predict': predicted[i].cpu().detach().tolist()})
            outpre[filename[i]].update({'prob': probs[i].cpu().detach().data.numpy().tolist()})
    
    if config['task'] == 'taskA':
        allscore = score_computer.compute_scoreA(outpre)
    elif config['task'] == 'taskB':
        allscore = score_computer.compute_scoreB(outpre)

    lossmean = np.mean(np.array(lossvec))
    return model, lossmean, allscore, outpre


def processing(config, mconfig, dataset=None):
    """
    Train a Hate-Attention with early stopping and evaluate on the test set.

    Parameters
    ----------
    config : dict
        Configuration.
    mconfig : dict
        Configuration for the model
    dataset : optional
        Object providing `train_dataset`, `val_dataset`, and `test_dataset` splits.

    Side Effects
    ------------
    - Writes a `train.log` file under `config["modeldir"]`.
    - Saves per-epoch predictions as JSON in `config["resultsdir"]`.
    - Saves the best model as `bestmodel.pkl` in `config["modeldir"]`.
    - Writes final test predictions as `besttestf1_*.json` in `config["resultsdir"]`.

    Returns
    -------
    None
        The function logs metrics and writes outputs to disk; it does not return a value.
    """
    # parameters for output dimension and loss function
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

    # parameters for early stoping
    evalacc_best = 0
    early_wait = 5
    run_wait = 1
    continuescore = 0
    stop_counter = 0

    # other settings
    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]
    logging.basicConfig(filename=os.path.join(config["modeldir"], 'train.log'), level=logging.INFO)

    # create model
    model = CLIP_multi(odim=odim, modelname=config["MODELtext"], cachedir=config["cachedir"], mconfig=mconfig)
    
    # use multiple GPUs if it is available
    # it is super easy to use (just a wrapper).
    # but it is Bottleneck on the main GPU (becomes overloaded).
    # slower communication (uses Python threads, not optimized).
    # less scalable → fine for 2 GPUs, but poor for many GPUs.
    # need change to DistributedDataParallel (DDP) 
    model.to(config['device'])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    
    # compute statistic information of the model and dataset, save them in the log file
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The model has {trainable_parameters} trainable parameters')
    logging.info(f'Train set contains {len(dataset.train_dataset)} samples')
    logging.info(f'Dev set contains {len(dataset.val_dataset)} samples')
    logging.info(f'Test set contains {len(dataset.test_dataset)} samples')

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  weight_decay=0.0001)
    
    # create data loader for the train, validation and test set
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
        
        # train the model
        model.train()
        model, lossmean, allscore, outpre = run(data_loader_train, model, config, criterion, train=True, optimizer=optimizer)
        logging.info(f'Epoch {epoch} training loss is {lossmean}, train weighted F1 score is {allscore}.')


        # evaluate the model
        print('evaluation')
        torch.cuda.empty_cache()
        model.eval()
        model, lossmean, allscore, outpre = run(data_loader_dev, model, config, criterion, train=False)
        logging.info(f'Epoch {epoch} evaluation loss is {lossmean}, evaluation weighted F1 score is {allscore}.')

        # save checkpoint model and intermediate results with early stoping
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,'bestmodel.pkl')

        with open(os.path.join(resultsdir, str(epoch) + '_' + str(lossmean) + '_' + str(
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

        if continuescore >= run_wait:
            stop_counter = 0
        print(stop_counter)
        print(early_wait)
        if stop_counter < early_wait:
            pass
        else:
            break

    # load the best performance model and evaluate on the test set
    model = torch.load(os.path.join(modeldir,'bestmodel.pkl'), map_location=config["device"])
    model.eval()
    _, _, allscore, outpre = run(data_loader_test, model, config, criterion, train=False)
    testscore = float(allscore)
    logging.info(f'Test weighted F1 score is {testscore}.')
    with open(os.path.join(resultsdir, 'besttestf1_' + str(testscore)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)


def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./dataset', type=str, help='dir saves the processed data')
    parser.add_argument('--modal', default='Hate-Attention', type=str, help='model type')
    parser.add_argument('--task', default='taskA', type=str, help='which task, options: taskA, taskB')
    parser.add_argument('--savedir', default='./output', type=str, help='dir saves the trained model and results')
    parser.add_argument('--cashedir', default='./CASHE', type=str, help='dir saves downloaded pre-trained language models')
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

    modeldir = os.path.join(savedir, 'Multimodal', modal, task, 'model')                    # dir saves the trained model
    resultsdir = os.path.join(savedir, 'Multimodal', modal, task, 'results')                # dir saves the predicted test set results

    for makedir in [modeldir, resultsdir, cashedir]:                                        # create the folders if it not exist
        if not os.path.exists(makedir):
            os.makedirs(makedir)

    max_num_epochs = 100                                                                    # define the maximum number of epochs

    if torch.cuda.is_available() == True:                                                   # set GPU if it is aviliable
        device = 'cuda'
    else:
        device = 'cpu'

    # load data from the generated JSON files for the train, validation and test sets
    with open(os.path.join(datadir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    with open(os.path.join(datadir, "val.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)

    # the pre-trained CLIP model used
    MODELtext = "openai/clip-vit-large-patch14"  

    # load the pre-trained CLIP model text and image tokenizer
    image_processor = CLIPImageProcessor.from_pretrained(MODELtext, cache_dir=cashedir)
    tokenizer = CLIPTokenizer.from_pretrained(MODELtext, cache_dir=cashedir)

    dataset = CLIPdatasetclass(train_file=traindict,                                        # create dataset for training and evaluation
                               val_file=valdict,
                               test_file=testdict,
                               device=device,
                               task=task)

    # model configuration
    mconfig = {"d_model": 1024, #1024,
               "n_block": 2,
               "n_head": 8,
               "task": task
    }
    # Configguration for training and evaluation
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
    # train and evaluate model
    processing(config, mconfig, dataset)








