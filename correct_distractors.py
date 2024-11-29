import transformers
import torch
import argparse
import json
import os
import copy
import time
from transformers import AutoTokenizer
from huggingface_hub import login
import re
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import random

class ComputeResults:
    def __init__(self, args):
        """
        Initializes the class with the model type and the method you want to use (API or pipeline).

        :param model_name: Name of the model to use.
        :param use_api: If True, uses the Ollama API; if False, uses the transformers pipeline.
        :param token: Hugging Face token (only required if using the llama model).
        """
        self.token = args.token
        self.output_dir = args.output_dir
        self.pipeline = self.load_pipeline()
        self.lemmatizer = WordNetLemmatizer()
        self.batch_prompts = None

        nltk.download('punkt', download_dir=self.output_dir)
        nltk.download('wordnet')
        nltk.download('punkt_tab')
	    
    def load_pipeline(self):
        """
        Loads the transformers pipeline for the especified model.
        """
        # Login to Hugging Face
        assert self.token != None, "Unable to login Hugging Face, please provide a HF token."
        login(self.token)

        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
            
        return transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            pad_token_id=50256,
        )


    def create_prompt(self, question, correct_answer):
        return [
            {"role": "system", "content": "You are an assistant that generates plausible but incorrect distractors for a multiple-choice question."},
            {"role": "user", "content": (
                f"Question: {question}\n"
                f"Correct Answer: {correct_answer}\n"
                "Generate four plausible but incorrect answers (distractors) for the question above. "
                "The distractors must:\n"
                "- Be relevant to the question.\n"
                "- Be short and simple, similar in length to the correct answer.\n"
                "- Clearly not be the correct answer.\n"
                "- Not be synonymous with or too closely related to the correct answer.\n"
                "The output should:\n"
                "- Contain exactly four distractors.\n"
                "- Be concise, with no additional explanation or context.\n"
                "- Be formatted as follows:\n"
                "1. Distractor 1\n"
                "2. Distractor 2\n"
                "3. Distractor 3\n"
                "4. Distractor 4\n"
                "Do not include any extra text or commentary, just the four distractors."
            )},
        ]


    def generate(self, annotation):
        """
        Main method for result generation. Depending on the initial configuration uses the API or the pipeline.
        """
        prompt = self.create_prompt(annotation['question'], annotation['correct_choice'])
        output = self.pipeline(prompt, max_new_tokens=500, truncation=True)
        generated_text = output[0]['generated_text']
        result, answer = self.get_result(generated_text)
        processed_result = self.result_processing(result)
        processed_result.append(answer)
        return processed_result
    

    def get_result(self, generated_text):
        """
        Processes the generated text to get the final result and the original answer.
        """
        user_text = generated_text[1]['content']  
        return generated_text[-1]['content'], user_text.split("Answer: ")[1].split("\n")[0]

    
    def result_processing(self, result, word_limit=8):
        lines = result.splitlines()
        options = [line.split('. ', 1)[1].strip() for line in lines if '. ' in line]
        return [self.normalize_answer(option) for option in options]


    def normalize_answer(self, answer):
        answer = answer.lower()
        answer = re.sub(r'[^a-zA-Z0-9\s]', '', answer)

        # Tokenize and lematize
        words = nltk.word_tokenize(answer)
        lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
        
        normalized_answer = ' '.join(lemmatized)
        return normalized_answer
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="/gaueko0/users/ietxarri010/GrAL_Irene/okvqa", help="Path to the OkVqa prediction files."
    )
    parser.add_argument(
        "--output_dir", type=str, default="'/gaueko0/users/ietxarri010/nltk_data'", help="Path to the OkVqa prediction files."
    )
    parser.add_argument(
        "--token", type=str, default=None, help="HuggingFace login token"
    )
    args = parser.parse_args()
    return args
    
def main():

    print(f"transformers version: {transformers.__version__}")
    print("Parsing args...")
    args = parse_args()
    print(f'Args parsed.')
    
    # Create class instances
    compute = ComputeResults(args)

    # Load json files
    print("\nLoading json files...")
    with open(os.path.join(args.root, 'train', f'annotations_train_mc_distractors.json'), "r") as f: 
      train = json.load(f)
    with open(os.path.join(args.root, 'train', f'annotations_train_mc_distractors_llama3.1_8b.json'), "r") as f: 
      train_syn = json.load(f)
    with open(os.path.join(args.root, 'val', f'annotations_val_mc_distractors.json'), "r") as f: 
      val = json.load(f)
    with open(os.path.join(args.root, 'test', f'annotations_test_mc_distractors.json'), "r") as f: 
      test = json.load(f)
    print(f'Files loaded.')


    def correct_choices (annotations, wrongs):
        for i in wrongs:
            ann = annotations[i]
            ann['choices'] = compute.generate(ann)


    def count_wrongs (annotations):
        incorrect = []
        for i, ann in enumerate(annotations):
            if len(ann['choices']) < 5: incorrect.append(i)
        return incorrect


    def check_annotations (annotations):
        incorrect = count_wrongs (annotations)
        size = len(incorrect)
        update = True
        while size > 0 and update:
            correct_choices(annotations, incorrect)
            incorrect = count_wrongs (annotations)
            update = size != len(incorrect)
            size = len(incorrect)
        if size >0:
            print(f'UNCHANGED! wrongs:')
            '\n'.join(incorrect)
            print(incorrect)
            

    def shuffle_annotations(annotations):
        for ann in annotations:
            random.shuffle(ann['choices'])
            ann['correct_choice_idx'] = ann['choices'].index(ann["correct_choice"])


    check_annotations(train['annotations'])
    print('shuffling...')
    shuffle_annotations(train['annotations'])
    print('train finished!')
    print('-------------------\n')

    check_annotations(train_syn['annotations'])
    print('shuffling...')
    shuffle_annotations(train_syn['annotations'])
    print('train_syn finished!')
    print('-------------------\n')

    check_annotations(val['annotations'])
    print('shuffling...')
    shuffle_annotations(val['annotations'])
    print('val finished!')
    print('-------------------\n')

    check_annotations(test['annotations'])
    print('shuffling...')
    shuffle_annotations(test['annotations'])
    print('test finished!')
    print('-------------------\n')

    # Update json files
    print("\nUpdating json files...")
    with open(os.path.join(args.root, 'train', f'annotations_train_mc_distractors.json'), "w") as f: 
      json.dump(train, f, indent=4)
    with open(os.path.join(args.root, 'train', f'annotations_train_mc_distractors_llama3.1_8b.json'), "w") as f: 
      json.dump(train_syn, f, indent=4)
    with open(os.path.join(args.root, 'val', f'annotations_val_mc_distractors.json'), "w") as f: 
      json.dump(val, f, indent=4)
    with open(os.path.join(args.root, 'test', f'annotations_test_mc_distractors.json'), "w") as f: 
      json.dump(test, f, indent=4)
    print(f'Files updated.')

if __name__ == "__main__":
    main()