from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os, json

with open(os.path.join("dataset/Memotion3", "train_en.json"), encoding="utf8") as json_file:
    traindict = json.load(json_file)
with open(os.path.join("dataset/Memotion3", "test_en_labels.json"), encoding="utf8") as json_file:
    testdict = json.load(json_file)
with open(os.path.join("dataset/Memotion3", "val_en.json"), encoding="utf8") as json_file:
    valdict = json.load(json_file)
valdict = {str(key) + '_dev': val for key, val in valdict.items()}

traindict.update(valdict)

for dset, dsetdict in zip(['train', 'test'], [traindict, testdict]):
    text = []
    for i in list(dsetdict.keys()):
        text.append(dsetdict[i]["text"])
    text = ' '.join(text)


    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the generated word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axes
    plt.savefig('./output/Memotion3/' + dset + '_wordcloud.pdf')


