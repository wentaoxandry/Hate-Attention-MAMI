import os, json
import random
import numpy as np
from tqdm import tqdm
import operator
from collections import defaultdict
from PIL import Image
import cv2
from skimage import feature
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# this need java, which server didn't installed
import language_tool_python
tool = language_tool_python.LanguageTool('en-US')

def cal_grammar_score(text):
    # language-tool-python package11 to assess the English grammaticality
    # score 𝐺 of the text,
    scores_word_based_sentence = []
    scores_sentence_based_sentence = []


    matches = tool.check(text)
    count_errors = len(matches)
    # only check if the sentence is correct or not
    scores_sentence_based_sentence.append(np.min([count_errors, 1]))
    scores_word_based_sentence.append(count_errors)
    word_count = len(text.split())
    sum_count_errors_word_based = np.sum(scores_word_based_sentence)
    score_word_based = 1 - (sum_count_errors_word_based / word_count)
    return score_word_based



def average_pixel_width(im):  
    # edginess, quantifies the informativeness of the vision
    # stream by calculating the percentage of edge pixels relative to the total number of pixels in the
    # image.
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100


def get_blurrness_score(path):
    # blurriness score 𝐵 of the image. The blurriness score is calculated as the variance of
    # the gray-scaled image after being filtered by a Laplace filter.
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./../../../dataset', type=str, help='Dir saves the datasource information')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    savedir = os.path.join(datadir, 'RMs')
    if not os.path.exists(savedir):
        os.makedirs(savedir)



    with open(os.path.join(datadir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    with open(os.path.join(datadir, "val.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)

    '''with open(os.path.join(datadir, "mami", "training.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "mami", "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    Ntest = len(testdict.keys())
    import random
    valdict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), Ntest))}
    for k in list(valdict.keys()):
        del traindict[k]
    with open(os.path.join(datadir, "mami", "training_sub.json"), 'w', encoding='utf-8') as f:
                json.dump(traindict, f, ensure_ascii=False, indent=4)
    with open(os.path.join(datadir, "mami", "val.json"), 'w', encoding='utf-8') as f:
                json.dump(valdict, f, ensure_ascii=False, indent=4)'''
    
    '''import random
    traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 100))}
    testdict = {k: testdict[k] for k in list(random.sample(list(testdict.keys()), 100))}
    valdict = {k: valdict[k] for k in list(random.sample(list(valdict.keys()), 100))}
'''
    for dset, workdict in zip(['train', 'dev', 'test'], [traindict, valdict, testdict]):
        textscoredict = {}
        imagescoredict = {}
        for key in tqdm(list(workdict.keys())):
            text = workdict[key]['text'].replace('\n', ' ')
            imgdir = workdict[key]['imagedir']#.replace('./', './../../../')
            image = Image.open(imgdir) 
            if image.mode != 'RGB':
                image = image.convert('RGB')

            average_pixel_width_score = average_pixel_width(image) / 100
            blurrness_score = get_blurrness_score(imgdir) / 5000
            imagescore = [average_pixel_width_score, blurrness_score]

            imagescoredict.update({key: imagescore})
            textscores = cal_grammar_score(text)
            textscoredict.update({key: textscores})

        with open(os.path.join(savedir, "text_" + dset + ".json"), 'w', encoding='utf-8') as f:
            json.dump(textscoredict, f, ensure_ascii=False, indent=4)
        with open(os.path.join(savedir, "image_" + dset + ".json"), 'w', encoding='utf-8') as f:
             json.dump(imagescoredict, f, ensure_ascii=False, indent=4)

    








