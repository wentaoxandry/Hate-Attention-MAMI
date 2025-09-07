# -*- coding: utf-8 -*-
import json
import os
from nltk.tokenize import TweetTokenizer
from ftfy import fix_text
from emoji import UNICODE_EMOJI
import argparse
from emoji import demojize
tokenizer = TweetTokenizer()

def normalizeTFToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return token.replace('@', '')
    elif token.startswith("#"):
        return token.replace('#', '')
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return 'URL'
    elif token in UNICODE_EMOJI['en']:
        return demojize(token)
    else:
        return token

def nospecial(text):
	import re
	text = re.sub("[^a-zA-Z0-9 .,?!\']+", "",text)
	return text

def text_process(text):
    texttrans = fix_text(text)  # some special chars like 'à¶´à¶§à·’ à¶»à·à¶½à·Š'
                                # will transformed into the right form පටි රෝල්
    tokens = tokenizer.tokenize(texttrans.replace('\n', ''))
    normTweet = " ".join(filter(None, [normalizeTFToken(token) for token in tokens]))

    normTweet = normTweet.replace(' ’ ', '’')
    normTweet = normTweet.replace(' .', '.')
    normTweet = normTweet.replace(' ,', ',')
    normTweet = normTweet.replace(' ?', '?')
    normTweet = normTweet.replace(' !', '!')

    texttrans = nospecial(normTweet)
    texttrans = texttrans.lower()
    return texttrans

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--sourcedir', default='./Sourcedata/MAMI', type=str, help='dir saves the downloaded raw data')
    parser.add_argument('--savedir', default='./dataset/MAMI', type=str, help='dir saves the processed data')
    parser.add_argument('--metadir', default='./meta_info/MAMI', type=str, help='dir save the meta information, which is used for the results reproduction')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    sourcedir = args.sourcedir  
    savedir = args.savedir
    metadir = args.metadir
    
    valdir = os.path.join(metadir, 'val.txt')                       # load validation set information
    my_file = open(valdir, "r") 
    data = my_file.read() 
    vallist = data.split("\n") 
    my_file.close() 
    vallist.remove('')


    if not os.path.exists(savedir):                                 # create folder to save the processed data
        os.makedirs(savedir)


    # 1. save normalized text, image path, task A and B ground truth labels in JSON file
    # 2. based on meta information spilit validation set from training set. 
    dsets = ['test', 'training']
    for dset in dsets:
        setsdir = os.path.join(sourcedir, dset, dset + '.csv')
        imagemaindir = os.path.join(sourcedir, dset)
        with open(setsdir, encoding="utf8") as f:
            data = f.readlines()
        del data[0]

        if dset == 'training':
            datadict = {}
            for i in data:
                split = i.split('\t')
                imagedir = os.path.join(imagemaindir, split[0])
                misogynous = split[1]
                taskB = [split[2], split[3], split[4], split[5]]

                texttrans = split[6].strip('\n')
                texttrans = text_process(texttrans)
                
                fileid = split[0] + '_' + dset
                datadict.update({fileid: {}})
                datadict[fileid].update({'imagedir': imagedir})
                datadict[fileid].update({'taskA': misogynous})
                datadict[fileid].update({'taskB': taskB})
                datadict[fileid].update({'text': texttrans})

            valdict = {}
            traindict = {}
            for i in vallist:
                valdict.update({i + '_training': datadict[i + '_training']})
            for i in list(datadict.keys()):
                if i.split('_')[0] in vallist:
                    pass
                else:
                    traindict.update({i: datadict[i]})
            with open(os.path.join(savedir, "train.json"), 'w', encoding='utf-8') as f:
                json.dump(traindict, f, ensure_ascii=False, indent=4)
            with open(os.path.join(savedir, "val.json"), 'w', encoding='utf-8') as f:
                json.dump(valdict, f, ensure_ascii=False, indent=4)
        else:
            datadict = {}
            testlabeldir = os.path.join(sourcedir, dset, 'test_labels.txt')
            testlabel = {}
            with open(testlabeldir) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('\t')
                testlabel.update({line[0]: [line[1], line[2], line[3], line[4], line[5].strip('\n')]})
            for i in data:
                split = i.split('\t')
                imagedir = os.path.join(imagemaindir, split[0])
                texttrans = split[1].strip('\n')
                texttrans = text_process(texttrans)

                fileid = split[0] + '_' + dset
                datadict.update({fileid: {}})
                datadict[fileid].update({'imagedir': imagedir})
                datadict[fileid].update({'taskA': testlabel[split[0]][0]})
                datadict[fileid].update({'taskB': testlabel[split[0]][1:]})
                datadict[fileid].update({'text': texttrans})

            with open(os.path.join(savedir, dset + ".json"), 'w', encoding='utf-8') as f:
                json.dump(datadict, f, ensure_ascii=False, indent=4)




