import transformers
import torch
import argparse
import json
import os
import copy
from huggingface_hub import login
#from progressbar import ProgressBar


def create_prompt(question, answer, is_llama=False):
    if is_llama:
        return [
            {"role": "system", "content": "You are an assistant that provides rephrased answers using one-word synonyms or a few words equivalent term."},
            {"role": "user", "content": f"Question: {question}\n"
                                        f"Answer: {answer}\n"
                                        "If the answer is a proper noun, keep it unchanged. "
                                        "If you can confidently provide a one-word synonym or equivalent term, do so. "
                                        "Otherwise, leave the answer unchanged. "
                                        "Do not include any additional text or explanation."},
        ]
    else:
        return f"Question: {question}\n" \
               f"Answer: {answer}\n" \
               "If the answer is a proper noun, keep it unchanged. " \
               "If you can confidently provide a one-word synonym or equivalent term, do so. " \
               "Otherwise, leave the answer unchanged. Do not include any additional text or explanation."


def get_batch_prompts(annotations, model_type):
    batch_prompts = []
    is_llama = "llama" in model_type  
    for ann in annotations:
        question = ann["question"]
        answers = [str(ans["answer"]) for ans in ann["answers"]]
        for answer in answers:
            prompt = create_prompt(question, answer, is_llama)
            batch_prompts.append(prompt)
    return batch_prompts


def result_processing(result, answer, word_limit=8):
    if len(result.split()) > word_limit:
        return answer
    return result


def compute_results(pipeline, batch_prompts, batch_size=16):
    results = []

    for i in range(0, len(batch_prompts), batch_size):
        batch = batch_prompts[i:i+batch_size]
        outputs = pipeline(batch, max_new_tokens=10, truncation=True)

        for output in outputs:
            generated_texts = output[0]['generated_text']
            user_text = generated_texts[1]['content'] 
            answer = user_text.split("Answer: ")[1].split("\n")[0]
            result = generated_texts[-1]['content']  
            
            processed_result = result_processing(result, answer)
            results.append(processed_result)

    return results


def replace_info(data, results):
  data_ = copy.deepcopy(data)
  for idx, ann in enumerate(data["annotations"]):
    for answer_id in range(len(ann["answers"])):
      result = results.pop(0)
      data_["annotations"][idx]["answers"][answer_id]["answer"] = result
  return data_


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["openchat", "llama-8b", "llama-70b"],
        help="Model type to be fine-tuned."
    )
    parser.add_argument(
        "--root", type=str, default="/gaueko0/users/ietxarri010/GrAL_Irene/okvqa", help="Path to the OkVqa prediction files."
    )
    parser.add_argument(
        "--token", type=str, default=None, help="HuggingFace login token"
    )
    args = parser.parse_args()
    return args


def main():
    print("Parsing args...")
    args = parse_args()

    """
    command = ['huggingface-cli', 'login', '--token', args.token, '--add-to-git-credential']
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print('Output:', result.stdout)
    except subprocess.CalledProcessError as e:
        print('Error:', e.stderr)
    """

    # Load jsons
    with open(os.path.join(args.root, 'train', f'annotations_train.json'), "r") as f: 
      train = json.load(f)
      #train["annotations"] = train["annotations"][:5]
    with open(os.path.join(args.root, 'val', f'annotations_val.json'), "r") as f: 
      val = json.load(f)
      #val["annotations"] = val["annotations"][:5]

    # Get prompts
    train_batch_prompts = get_batch_prompts(train["annotations"], args.model_type)
    val_batch_prompts = get_batch_prompts(val["annotations"], args.model_type)
    
    # Load pipeline
    if "llama" in args.model_type:
        assert args.token != None, "Unable to login Hugging Face, please provide a HF token." 
        login(args.token) 

        if args.model_type == "llama-8b":
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                pad_token_id=50256, #eos
            )

    else: #openchat
        pipeline = transformers.pipeline(
            "text-generation", 
            model="openchat/openchat-3.5-0106", 
            torch_dtype=torch.bfloat16, device_map="auto"
        )

        """
        for idx, ann in enumerate(data["annotations"][:5]):
            question = ann["question"]
            answers = [str(ans["answer"]) for ans in ann["answers"]]
            for answer in answers:
                #prompt = f"Question: {question}\nAnswer: {answer}\nPlease rewrite the answer using synonyms or rephrasing."
                #prompt = f"Question: {question}\nAnswer: {answer}\nPlease replace '{answer}' with a synonym or an equivalent term."
                prompt = f"Please replace '{answer}' with a synonym or an equivalent term."
                result = pipeline(prompt, max_length=20, truncation=True, num_return_sequences=1) 
                text = result[0]['generated_text']
        """
    

    # Train 
    train_results = compute_results(pipeline, train_batch_prompts)
    train_ = replace_info(train, train_results)
    train_path = os.path.join(args.root, 'train', 'annotations_train_.json')
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, 'w') as json_file:
        json.dump(train_, json_file, indent=4)
    print("Train set finnished!\n")

    # Val 
    val_results = compute_results(pipeline, val_batch_prompts)
    val_ = replace_info(val, val_results)
    val_path = os.path.join(args.root, 'val', 'annotations_val_.json')
    os.makedirs(os.path.dirname(val_path), exist_ok=True)
    with open(val_path, 'w') as json_file:
        json.dump(val_, json_file, indent=4)
    print("Val set finnished!\n")


if __name__ == "__main__":
    main()