import os, argparse
import json
import openai
import base64

# Replace 'your_openai_api_key' with your actual OpenAI API key
key_dir = '/home/username/project/keys/openai_api_key.txt'
file = open(key_dir, "r")
openai.api_key = file.read()


def encode_image(image_path):
    """
    Encode an image file to a base64 string.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    str
        Base64-encoded image content.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def analyze_meme(image_path):
    """
    Run classification and captioning for a meme image via OpenAI.

    Parameters
    ----------
    image_path : str
        Path to the meme image to analyze.

    Returns
    -------
    (str, str)
        A tuple of (posteriors, caption), where:
        - posteriors is the raw string returned by the classifier prompt
        - caption is the raw string returned by the captioning prompt
    """
    encoded_image = encode_image(image_path)
    classification_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "First misogynous meme identification. The type of misogynous meme should be further recognized as misogynous related stereotype, shaming, objectification and violence (one sample may belong to more than 2 classes). Return the posteriors of misogynous, stereotype, shaming, objectification and violence with 4 decimals as a LIST. Do not include any explanation."},
                    {"image": encoded_image},
                ]
            },
        ],
        max_tokens=30
    )

    caption_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": "describe image with web knowledge. Without returning text captions, websites, sources, or intimation. All within 40 words."},
                    {"image": encoded_image},
                ]
            },
        ],
        max_tokens=100
    )


    posteriors = classification_response.choices[0].message.content
    caption = caption_response.choices[0].message.content

    return posteriors, caption


# Analyze all memes in the specified folder
def analyze_memes_in_folder(folder_path):
    """
    Analyze all meme images in a folder.

    Parameters
    ----------
    folder_path : str
        Directory containing images (.png, .jpg, .jpeg).

    Returns
    -------
    dict
        Mapping from filename to a result dict with keys:
        - 'prob' : str, the classifier output
        - 'caption' : str, the captioning output
    """
    results = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            
            image_path = os.path.join(folder_path, filename)
            try:
                posteriors, caption = analyze_meme(image_path)
                results.update({filename: {}})
                results[filename].update({'prob': posteriors})
                results[filename].update({'caption': caption})
                print(filename)
                print(posteriors)
                print(caption)
            except:
                pass
    return results


# Save results to a JSON file
def save_results_to_json(results, output_path):
    """
    Save analysis results to a JSON file.

    Parameters
    ----------
    results : dict
        Results mapping as returned by `analyze_memes_in_folder`.
    output_path : str
        Path to the output JSON file.
    """
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)


def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--sourcedir', default='./Sourcedata/MAMI', type=str, help='path to raw data.')
    parser.add_argument('--modal', default='ChatGPT', type=str, help='model type')
    parser.add_argument('--savedir', default='./output', type=str, help='dir saves the trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    sourcedir = args.sourcedir
    savedir = args.savedir
    modal = args.modal

    captiondir = os.path.join(savedir, 'Multimodal', modal, 'zero_shot')                        # dir saves image caption results
    if not os.path.exists(captiondir):                                                          # create the folders if it not exist
        os.makedirs(captiondir)     

    for dset in ["training", "test"]:                                 
        results = analyze_memes_in_folder(os.path.join(sourcedir, dset))
        save_results_to_json(results, os.path.join(captiondir, dset + '.json'))

