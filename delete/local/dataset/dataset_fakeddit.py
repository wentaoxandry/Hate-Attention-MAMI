import os, json
import multiprocessing as mp
import pandas as pd
import argparse
from tqdm import tqdm
from PIL import Image
def checkIfImageIsAvaliable(imgdir):
    if (os.path.isfile(imgdir)):
        try:
            im = Image.open(imgdir)
            return True
        except OSError:
            return False
    else:
        return False

def createdict(item):
    datadict = {}
    #print(item['id'])

    imgdir = os.path.join(imagefiledir, item['id'] + '.jpg')
    try:
        imageexist = checkIfImageIsAvaliable(imgdir)
        if imageexist is True:
            datadict.update({item['id']: {}})
            datadict[item['id']].update({'title': item['clean_title']})
            datadict[item['id']].update({'imgdir': imgdir})
            datadict[item['id']].update({'2_way_label': item['2_way_label']})
            datadict[item['id']].update({'3_way_label': item['3_way_label']})
            datadict[item['id']].update({'6_way_label': item['6_way_label']})
        else:
            print('no image found')
    except:
        pass

    return datadict


def product_helper(args):
    return createdict(*args)

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--sourcedir', default='./Sourcedata/Fakeddit', type=str, help='Dir saves the Fakeddit dataset')
    parser.add_argument('--savedir', default='./dataset/Fakeddit', type=str, help='Dir saves metainformation')
    parser.add_argument('--ifmulticore', default=True, type=bool, help='If use multi processor to faster the process')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #### This script creats the dataset meta information in JSON files.
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasourcedir = args.sourcedir
    savedir = args.savedir
    ifmulticore = args.ifmulticore

    textfiledir = os.path.join(datasourcedir, 'downloaded', 'labels')
    global imagefiledir
    imagefiledir = os.path.join(datasourcedir, 'public_image_set')

    if not os.path.exists(savedir):
        os.makedirs(savedir)


    # Path to train data
    path_to_train_tsv = os.path.join(textfiledir, "multimodal_train.tsv")
    print("Path to train.tsv is: " + path_to_train_tsv)


    # Path to test data
    path_to_test_tsv = os.path.join(textfiledir, "multimodal_test_public.tsv")
    print("Path to test.tsv is: " + path_to_test_tsv)

    # Path to val data
    path_to_val_tsv = os.path.join(textfiledir, "multimodal_validate.tsv")
    print("Path to val.tsv is: " + path_to_val_tsv)



    # Excerpt from train set
    df_train_original = pd.read_csv(path_to_train_tsv, header=0, sep='\t')
    df_train_original = df_train_original.loc[:, ~df_train_original.columns.str.contains('^Unnamed')]

    # Excerpt from test set
    df_test_original = pd.read_csv(path_to_test_tsv, header=0, sep='\t')
    df_test_original = df_test_original.loc[:, ~df_test_original.columns.str.contains('^Unnamed')]

    # Excerpt from val set
    df_val_original = pd.read_csv(path_to_val_tsv, header=0, sep='\t')
    df_val_original = df_val_original.loc[:, ~df_val_original.columns.str.contains('^Unnamed')]
    df_val_original['title'] = df_val_original['title'].astype(str)

    for dsetdata, dset in zip([df_val_original, df_test_original, df_train_original], ['val', 'test', 'train']):
        dict_data = dsetdata.to_dict('records')
        savedict = {}
        results = []
        if ifmulticore is True:
            pool = mp.Pool()
            results.extend(pool.map(createdict, dict_data))
        else:
            for i in dict_data:
                results.append(createdict(i))
        newresults = []
        newresults = [i for i in results if i != {}]
        for i in newresults:
            savedict.update(i)
        with open(os.path.join(savedir, dset + '.json'), "w", encoding="utf-8") as f:
            json.dump(savedict, f, ensure_ascii=False, indent=4)






