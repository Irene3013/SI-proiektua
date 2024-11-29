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

    def get_batch_prompts(self, annotations):
        batch_prompts = []
        for ann in annotations:
            question = ann["question"]
            answers = [str(ans["answer"]) for ans in ann["answers"]]
            correct_answer = self.choose_answer(answers)
            prompt = self.create_prompt(question, correct_answer)
            batch_prompts.append(prompt)
        return batch_prompts


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


    def choose_answer(self, answers):
            counts = Counter(answers)
            for threshold in [3, 2]: 
                candidates = [ans for ans, count in counts.items() if count >= threshold]
                if candidates:
                    return random.choice(candidates)
            
            return random.choice(answers)

    def generate(self, annotations, batch_size=32):
        """
        Main method for result generation. Depending on the initial configuration uses the API or the pipeline.
        """
        results = []
        self.batch_prompts = self.get_batch_prompts(annotations)
        for i in range(0, len(self.batch_prompts), batch_size):
            batch = self.batch_prompts[i:i+batch_size]
            outputs = self.pipeline(batch, max_new_tokens=500, truncation=True)
            for output in outputs:
                generated_text = output[0]['generated_text']
                result, answer = self.get_result(generated_text)
                processed_result = self.result_processing(result)
                processed_result.append(answer)
                results.append(processed_result)
        return results
    

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



class Convert2MC:
    def __init__(self):
        self.json_info = None
        

    def get_json_info (self, split):
        return { "license": {
            "url": "http://creativecommons.org/licenses/by/4.0/",
            "name": "Creative Commons Attribution 4.0 International License"
          },
          "data_subtype": f"{split}2014",
          "question_types": {
            "eight": "Plants and Animals",
            "nine": "Science and Technology",
            "four": "Sports and Recreation",
            "six": "Geography, History, Language and Culture",
            "two": "Brands, Companies and Products",
            "other": "Other",
            "one": "Vehicles and Transportation",
            "five": "Cooking and Food",
            "ten": "Weather and Climate",
            "seven": "People and Everyday life",
            "three": "Objects, Material and Clothing"
          },
          "annotations": [],
          "info": {
            "year": 2019,
            "version": "1.0",
            "description": "This is v1.0 of the OK-VQA dataset."
          },
          "data_type": "mscoco" }


    def add_annotation (self, choices, annotation):
        correct_choice = choices[-1]
        random.shuffle(choices)    

        new_annotation = {
              "image_id": annotation ['image_id'],
              "question_id": annotation ['question_id'],
              "question_type": annotation ['question_type'],
              "question": annotation ['question'],
              "answer_type": annotation ['answer_type'],
              "choices": choices,
              "correct_choice_idx": choices.index(correct_choice),
              "correct_choice": correct_choice,
              "confidence": annotation ['confidence']
        }
        self.json_info["annotations"].append(new_annotation)


    def convert_MC (self, annotations, choices_list, split):
        self.json_info = self.get_json_info(split)
        for choices, annotation in zip(choices_list, annotations):
            self.add_annotation (choices, annotation) 
        return self.json_info

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
    convert = Convert2MC()

    # Load json files
    print("\nLoading json files...")
    with open(os.path.join(args.root, 'train', f'annotations_train.json'), "r") as f: 
      train = json.load(f)
    with open(os.path.join(args.root, 'train', f'annotations_train_llama3.1_8b.json'), "r") as f: 
      train_syn = json.load(f)
    with open(os.path.join(args.root, 'val', f'annotations_val.json'), "r") as f: 
      val = json.load(f)
    with open(os.path.join(args.root, 'test', f'annotations_test.json'), "r") as f: 
      test = json.load(f)
    print(f'Files loaded.')

    # Compute train  
    results = compute.generate(train["annotations"])
    data = convert.convert_MC(train["annotations"], results, 'train')
    path = os.path.join(args.root, 'train', f'annotations_train_mc_distractors.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print('Train saved!')

    # Compute train_syn  
    results = compute.generate(train_syn["annotations"])
    data = convert.convert_MC(train_syn["annotations"], results, 'train')
    path = os.path.join(args.root, 'train', f'annotations_train_mc_distractors_llama3.1_8b.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print('Train syn saved!')

    # Compute val  
    results = compute.generate(val["annotations"])
    data = convert.convert_MC(val["annotations"], results, 'val')
    path = os.path.join(args.root, 'val', f'annotations_val_mc_distractors.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print('Val saved!')

    # Compute test  
    results = compute.generate(test["annotations"])
    data = convert.convert_MC(test["annotations"], results, 'test')
    path = os.path.join(args.root, 'test', f'annotations_test_mc_distractors.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print('Test saved!')

    return 0

if __name__ == "__main__":
    main()
