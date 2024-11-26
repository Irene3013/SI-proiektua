import transformers
import torch
import argparse
import json
import os
import copy
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import re
import nltk
from nltk.stem import WordNetLemmatizer
from collections import Counter
import random


def create_prompt(question, correct_answer):
    return [
        {"role": "system", "content": "You are an assistant that generates plausible but incorrect distractors for a multiple-choice question."},
        {"role": "user", "content": f"Question: {question}\n"
                                    f"Correct Answer: {correct_answer}\n"
                                    "Generate four plausible incorrect answers (distractors) that could be used in a multiple-choice question. "
                                    "Make sure the distractors are relevant to the question and similar in length or complexity to the correct answer. "
                                    "The distractors should not be the same as the correct answer. "
                                    "Do not include any additional text or explanation, just provide the distractors as a list of four options."},
    ]


def choose_answer(answers):
        counts = Counter(answers)
        for threshold in [3, 2]: 
            candidates = [ans for ans, count in counts.items() if count >= threshold]
            if candidates:
                return random.choice(candidates)
        return random.choice(answers)


def get_batch_prompts(annotations):
    batch_prompts = []
    for ann in annotations:
        question = ann["question"]
        answers = [str(ans["answer"]) for ans in ann["answers"]]
        correct_answer = choose_answer(answers)
        prompt = create_prompt(question, correct_answer)
        batch_prompts.append(prompt)
    return batch_prompts


def replace_info(data, results):
    data_ = copy.deepcopy(data)
    for idx, ann in enumerate(data["annotations"]):
        for answer_id in range(len(ann["answers"])):
            result = results.pop(0)
            data_["annotations"][idx]["answers"][answer_id]["answer"] = result
    return data_

class ComputeResults:
    def __init__(self, token=None):
        """
        Initializes the class with the model type and the method you want to use (API or pipeline).

        :param model_name: Name of the model to use.
        :param use_api: If True, uses the Ollama API; if False, uses the transformers pipeline.
        :param token: Hugging Face token (only required if using the llama model).
        """
        self.token = token
        self.pipeline = self.load_pipeline()
        self.lemmatizer = WordNetLemmatizer()

        nltk.download('punkt', download_dir='/gaueko0/users/ietxarri010/nltk_data')
        nltk.download('wordnet')
	    

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


    def generate(self, batch_prompts, batch_size=32):
        """
        Main method for result generation. Depending on the initial configuration uses the API or the pipeline.
        """
        results = []

        for i in range(0, len(batch_prompts), batch_size):
            batch = batch_prompts[i:i+batch_size]

            outputs = self.pipeline(batch, max_new_tokens=10, truncation=True)

            for output in outputs:
                print(output)
                generated_text = output[0]['generated_text']
                result, answer = self.get_result(generated_text)
                processed_result = self.result_processing(result, answer)
                results.append(processed_result)
        return results
    
    def get_result(self, generated_text):
        """
        Processes the generated text to get the final result and the original answer.
        """
        user_text = generated_text[1]['content']  
        return generated_text[-1]['content'], user_text.split("Answer: ")[1].split("\n")[0]

    
    def result_processing(self, result, answer, word_limit=8):
        """
        Discards the longest results and replaces them with the orginal answer.
        """
        if len(result.split()) >= word_limit:
            return self.normalize_answer(answer)
        return self.normalize_answer(result) 


    def normalize_answer(self, answer):
        answer = answer.lower()
        answer = re.sub(r'[^a-zA-Z\s]', '', answer)

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
        "--token", type=str, default=None, help="HuggingFace login token"
    )
    args = parser.parse_args()
    return args


def main():

    print(f"transformers version: {transformers.__version__}")
    print("Parsing args...")
    args = parse_args()
    print(f'Args parsed.')
    

    # Load json files
    print("\nLoading json files...")
    with open(os.path.join(args.root, 'train', f'annotations_train.json'), "r") as f: 
      train = json.load(f)
      train = train[:2]
    with open(os.path.join(args.root, 'val', f'annotations_val.json'), "r") as f: 
      val = json.load(f)
      val = val[:2]
    print(f'Files loaded.')

    # Get prompts
    print("\nGetting batch promps...")
    train_batch_prompts = get_batch_prompts(train["annotations"])
    val_batch_prompts = get_batch_prompts(val["annotations"])
    print(f'Batch promps computed. \n')

    # Create class instance
    compute = ComputeResults(
        token=args.token 
    )

    # Compute train results 
    print("\nGenerating train answers...")
    time1 = time.time()
    train_results = compute.generate(train_batch_prompts)
    time2 = time.time()
    print(f'Train answers generated. Time: {(time2-time1)//3600}h {((time2-time1)%3600)//60}min {int((time2-time1)%60)}s')

    # Compute validation results 
    print("\nGenerating validation answers...")
    val_results = compute.generate(val_batch_prompts)
    time3 = time.time()
    print(f'Validation answers generated. Time: {(time3-time2)//3600}h {((time3-time2)%3600)//60}min {int((time3-time2)%60)}s')

    # Replace annotations with new answers
    print("\nReplacing annotations answers...")
    train_ = replace_info(train, train_results)
    val_ = replace_info(val, val_results)
    print(f"Answers replaced.")

    print(f"\nSaving answers to json files...")

    # Save train annotations to json file
    model_name = 'llama3.1_8b'
    train_path = os.path.join(args.root, 'train', f'annotations_train_{model_name}.json')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, 'w') as json_file:
        json.dump(train_, json_file, indent=4)
    print(f"Train set saved!")

    # Save validation annotations to json file
    val_path = os.path.join(args.root, 'val', f'annotations_val_{model_name}.json')
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    with open(val_path, 'w') as json_file:
        json.dump(val_, json_file, indent=4)
    print("Val set saved!")
    

if __name__ == "__main__":
    main()
