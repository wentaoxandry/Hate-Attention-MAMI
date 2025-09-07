import os, argparse
import json
import openai

# Replace 'your_openai_api_key' with your actual OpenAI API key
key_dir = '/home/username/project/keys/openai_api_key.txt'
file = open(key_dir, "r")
openai.api_key = file.read()

def split_list(lst, chunk_size=50):
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

# ----------------------------
# Prompt blocks (exact wording)
# ----------------------------
EXAMPLES_HEADER = (
    "You are a linguistic expert. I will provide you with several text samples, each labeled as either misogynous or non-misogynous. "
    "If a text sample is classified as misogynous, it should be further categorized into sub-categories such as shaming, stereotyping, "
    "objectification, and violence (a single sample may belong to more than one sub-category).\n"
    "Here are the examples:\n"
)

INSTRUCTION_HEADER = (
    "Based on the above text samples and their labels, please classify the following samples. "
    "Return the results as a JSON-like dictionary in the following structure:\n\n"
    "{\n"
    "  \"filename\": {\"prob\": [misogynous, shaming, stereotyping, objectification, violence]},\n"
    "  ...\n"
    "}\n\n"
    "The five posterior probabilities should be rounded to four decimal places. Do not include any explanations.\n"
    "Here are the samples:\n"
)



def build_examples_block(train_data: dict) -> str:
    """
    EXACT example lines as in the screenshot, but with REAL captions and ground-truth probs from JSON.
    If 'prob' is missing, uses [] to keep the structure.
    OCR text not provided -> left empty to preserve the exact '+ ... + ...' separators.
    """
    lines = [EXAMPLES_HEADER]
    
    keylist = list(train_data.keys())
    for i, id in enumerate(keylist, 0):
        caption = train_data[id]['caption']

        label = [int(train_data[id]['taskA'][0])]
        labelB = train_data[id]['taskB']
        labelB = [int(i) for i in labelB]
        label.extend(labelB)

        text = train_data[id]['text']
        # EXACT shape: Text i: + example i OCR text + example i image captioning text + Label i: + example i ground truth
        lines.append(
            f"Text {i}: + {text} + Image caption {i}: + {caption} + Label {i}: + {label}"
        )
    return "\n".join(lines)

def build_instruction_block(testdict: dict, sub_test_list: list) -> str:
    lines = [INSTRUCTION_HEADER]

    for i, id in enumerate(sub_test_list, 0):
        caption = testdict[id]['caption']
        text = testdict[id]['text']

        # EXACT shape: Text i: + sample i OCR text + sample i image captioning text
        lines.append(
            f"Text {id}: + {text} + Image caption {id}: + {caption}"
        )
    return "\n".join(lines)

# ----------------------------
# Single-call, 50-sample classification
# ----------------------------
def classify_batch_50(examples_text, testdict, sub_test_list):
    instruction_text = build_instruction_block(testdict, sub_test_list)

    classification_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": examples_text + instruction_text}
                ]
            },
        ],
        max_tokens=2000
    )
    posteriors = classification_response.choices[0].message.content
    return posteriors




def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datadir', default='./dataset', type=str, help='dir saves the processed data')
    parser.add_argument('--modal', default='ChatGPT', type=str, help='model type')
    parser.add_argument('--savedir', default='./output', type=str, help='dir saves the trained model and results')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datadir = args.datadir
    savedir = args.savedir
    modal = args.modal

    fsdir = os.path.join(savedir, 'Multimodal', modal, 'few_shot')                        # dir saves image caption results
    if not os.path.exists(fsdir):                                                          # create the folders if it not exist
        os.makedirs(fsdir) 

    # load data from the generated JSON files for the train, validation and test sets
    with open(os.path.join(datadir, "train.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datadir, "test.json"), encoding="utf8") as json_file:
        testdict = json.load(json_file)

    # get 50 training samples as prior knowledge
    import random
    traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 100))}

    # load training set image caption
    with open(os.path.join(savedir, 'Multimodal', modal, 'zero_shot', "training.json"), encoding="utf8") as json_file:
        traincapdict = json.load(json_file)
    for i in list(traindict.keys()):
        traindict[i].update('caption': traincapdict[i]['caption'])
    
    # load test set image caption
    with open(os.path.join(savedir, 'Multimodal', modal, 'zero_shot', "test.json"), encoding="utf8") as json_file:
        testcapdict = json.load(json_file)
    for i in list(testdict.keys()):
        testdict[i].update('caption': testcapdict[i]['caption'])


    examples_text = build_examples_block(traindict)
    split_test_list = split_list(list(testdict.keys()), 50)

    output = {}
    for sub_test_list in split_test_list:
        # One prompt -> 50 results
        raw_output = classify_batch_50(examples_text, testdict, sub_test_list)
        output.update{raw_output}

    with open(os.path.join(fsdir, 'test.json'), 'w') as json_file:
        json.dump(output, json_file, indent=4)
