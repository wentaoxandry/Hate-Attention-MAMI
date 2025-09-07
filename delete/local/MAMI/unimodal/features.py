import os, json
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


def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

def perform_color_analysis(im, flag):
    #path = images_path + img 
    #im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    dark_percent = (dark_percent1 + dark_percent2)/2 
    
    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None

def average_pixel_width(im):  
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

def get_dominant_color(path):
    img = cv2.imread(path)
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    values, counts = np.unique(labels, return_counts=True)
    result = np.column_stack((values, counts))
    dominant_color = palette[np.argmax(result[:, -1])]
    return dominant_color

def get_blurrness_score(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./../../dataset', type=str, help='Dir saves the datasource information')
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



    with open(os.path.join(datadir, "mami", "training_sub.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "mami", "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    with open(os.path.join(datadir, "mami", "val.json"), encoding="utf8") as json_file:
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
    valdict = {k: valdict[k] for k in list(random.sample(list(valdict.keys()), 100))}'''

    for dset, workdict in zip(['train', 'dev', 'test'], [traindict, valdict, testdict]):
        textscoredict = {}
        imagescoredict = {}
        for nm, key in enumerate(tqdm(workdict.keys()), 0):
            text = workdict[key]['text'].replace('\n', ' ')
            imgdir = workdict[key]['imagedir']#.replace('./', './../../')
            image = Image.open(imgdir) 
            if image.mode != 'RGB':
                image = image.convert('RGB')

            dullness = perform_color_analysis(image, 'black') / 100
            whiteness = perform_color_analysis(image, 'white') / 100
            average_pixel_width_score = average_pixel_width(image) / 100
            domin_color = get_dominant_color(imgdir) / 255
            blurrness_score = get_blurrness_score(imgdir) / 5000
            if dullness == None:
                dullness = 0.0
            if whiteness == None:
                whiteness = 0.0
            imagescore = [dullness, whiteness, average_pixel_width_score, domin_color[0], 
                          domin_color[1], domin_color[2], blurrness_score]

            imagescoredict.update({key: imagescore})
            textscores = cal_grammar_score(text)
            textscoredict.update({key: textscores})

        with open(os.path.join(dsetsavedir, "text_" + dset + ".json"), 'w', encoding='utf-8') as f:
            json.dump(textscoredict, f, ensure_ascii=False, indent=4)
        with open(os.path.join(savedir, "image_" + dset + ".json"), 'w', encoding='utf-8') as f:
             json.dump(imagescoredict, f, ensure_ascii=False, indent=4)

    








