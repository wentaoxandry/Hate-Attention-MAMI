import os
import json
import csv
import argparse
import numpy as np
import googletrans
from googletrans import Translator

print(googletrans.LANGUAGES)
translator = Translator()

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def readcsv(sourcedir, savedir):
    imagedir = os.path.join(sourcedir, 'image')
    for dirs in [savedir]:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
    for dset in ['val', 'train']:
        datadict = {}
        csvdir = os.path.join(sourcedir, 'memotion3', dset + '.csv')
        dsetimagedir = os.path.join(imagedir, dset)
        with open(csvdir, newline='') as csvfile:
            readdata = csv.reader(csvfile, delimiter='\t')#, delimiter=' ', quotechar='|')
            data = []
            for row in readdata:
                data.append(row)
        data = data[1:]
        allsamples = len(data)
        current = 0
        for i in data:
            datadict.update({i[0]: {}})
            #imageurl = i[1]
            #img_data = requests.get(imageurl).content
            imgsavedir = os.path.join(dsetimagedir, i[0] + '.jpg')
            #with open(imgsavedir, 'wb') as handler:
            #    handler.write(img_data)
            datadict[i[0]].update({'imagedir': imgsavedir})
            i[7] = i[7].replace('\n', ' ')
            result = translator.translate(i[7])
            datadict[i[0]].update({'text': result.text})
            try:
                result = translator.translate(i[7])
                datadict[i[0]].update({'text': result.text})
            except:
                datadict[i[0]].update({'text': i[7]})
                print('Error translate')
            current = current + 1
            print(str(current) + ' / ' + str(allsamples))
            if i[6] == 'negative' or i[6] == 'very_negative':
                taskA_label = 0
            elif i[6] == 'neutral':
                taskA_label = 1
            elif i[6] == 'positive' or i[6] == 'very_positive':
                taskA_label = 2
            else:
                print('task1_label error')

            if i[2] == 'not_funny':
                taskB_humorous = 0
            else:
                taskB_humorous = 1

            if i[3] == 'not_sarcastic':
                taskB_sarcastic = 0
            else:
                taskB_sarcastic = 1

            if i[4] == 'not_offensive':
                taskB_offensive = 0
            else:
                taskB_offensive = 1

            if i[5] == 'not_motivational':
                taskB_motivational = 0
            else:
                taskB_motivational = 1

            if i[2] == 'not_funny':
                taskC_humorous = 0
            elif i[2] == 'funny':
                taskC_humorous = 1
            elif i[2] == 'very_funny':
                taskC_humorous = 2
            elif i[2] == 'hilarious':
                taskC_humorous = 3
            else:
                print('taskC_humorous error')

            if i[3] == 'not_sarcastic':
                taskC_sarcasm = 0
            elif i[3] == 'general':
                taskC_sarcasm = 1
            elif i[3] == 'twisted_meaning':
                taskC_sarcasm = 2
            elif i[3] == 'very_twisted':
                taskC_sarcasm = 3
            else:
                print('taskC_sarcasm error')

            if i[4] == 'not_offensive':
                taskC_offense = 0
            elif i[4] == 'slight':
                taskC_offense = 1
            elif i[4] == 'very_offensive':
                taskC_offense = 2
            elif i[4] == 'hateful_offensive':
                taskC_offense = 3
            else:
                print('taskC_offense error')

            if i[5] == 'not_motivational':
                taskC_motivation = 0
            elif i[5] == 'motivational':
                taskC_motivation = 1
            else:
                print('taskC_motivation error')

            datadict[i[0]].update({'taskA': taskA_label})
            datadict[i[0]].update({'taskB': {}})
            datadict[i[0]]['taskB'].update({'humorous': taskB_humorous})
            datadict[i[0]]['taskB'].update({'sarcastic': taskB_sarcastic})
            datadict[i[0]]['taskB'].update({'offensive ': taskB_offensive})
            datadict[i[0]]['taskB'].update({'motivation': taskB_motivational})

            datadict[i[0]].update({'taskC': {}})
            datadict[i[0]]['taskC'].update({'humorous': taskC_humorous})
            datadict[i[0]]['taskC'].update({'sarcastic': taskC_sarcasm})
            datadict[i[0]]['taskC'].update({'offensive': taskC_offense})
            datadict[i[0]]['taskC'].update({'motivation': taskC_motivation})

        ## Here is used to creat a 4 folder cross validation ids
        '''allids = list(datadict.keys())
        random.shuffle(allids)
        cvtestlists = list(divide_chunks(allids, int(len(allids) / 4)))
        for j in range(len(cvtestlists)):
            np.savetxt(os.path.join(savedir, 'cvtest' + str(j) + '.txt'), cvtestlists[j], delimiter=" ", fmt="%s")'''
        ## Use this code to read files
        '''with open(os.path.join(savedir, 'cvtest1.txt')) as f:
            lines = f.readlines()'''

        with open(os.path.join(savedir, dset + '_en.json'), 'w', encoding='utf-8') as f:
            json.dump(datadict, f, ensure_ascii=False, indent=4)

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--sourcedir', default='./Sourcedata/Memotion3', type=str, help='Dir saves the datasource information')
    parser.add_argument('--savedir', default='./dataset/Memotion3', type=str, help='which data stream')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    sourcedir = args.sourcedir
    savedir = args.savedir

    readcsv(sourcedir, savedir)

    with open(os.path.join(savedir, "test_en.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)
    testlabeldir = os.path.join(sourcedir, 'memotion3', 'test_answer.txt')
    test_file = open(testlabeldir, "r")
    data = test_file.read()
    data_into_list = data.replace('\n', ' ').split(" ")[:-1]
    test_file.close()

    for i in range(len(data_into_list)):
        labels = data_into_list[i].split('_')
        testdict[str(i)].update({'taskA': int(labels[0])})
        testdict[str(i)].update({'taskB': {}})
        testdict[str(i)]['taskB'].update({'humorous': labels[1][0]})
        testdict[str(i)]['taskB'].update({'sarcastic': labels[1][1]})
        testdict[str(i)]['taskB'].update({'offensive': labels[1][2]})
        testdict[str(i)]['taskB'].update({'motivation': labels[1][3]})

        testdict[str(i)].update({'taskC': {}})
        testdict[str(i)]['taskC'].update({'humorous': labels[2][0]})
        testdict[str(i)]['taskC'].update({'sarcastic': labels[2][1]})
        testdict[str(i)]['taskC'].update({'offensive': labels[2][2]})
        testdict[str(i)]['taskC'].update({'motivation': labels[2][3]})


    with open(os.path.join(savedir, 'test_en_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(testdict, f, ensure_ascii=False, indent=4)


