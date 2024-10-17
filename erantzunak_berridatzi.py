import transformers
import torch
import argparse
import json
import os

def get_llama_batch_prompts(annotations):
    batch_prompts = []
    for idx, ann in enumerate(annotations[:5]):
        question = ann["question"]
        answers = [str(ans["answer"]) for ans in ann["answers"]]
        for answer in answers:
            messages = [
                {"role": "system", "content": "You are an assistant that provides rephrased answers using one-word synonyms or a few words equivalent term."},
                {"role": "user", "content": f"Question: {question}\n"
                                            f"Answer: {answer}\n"
                                             "If the answer is a proper noun, keep it unchanged. "
                                             "If you can confidently provide a one-word synonym or equivalent term, do so. "
                                             "Otherwise, leave the answer unchanged. "
                                             "Do not include any additional text or explanation."},
                #{"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nIf a one-word synonym or equivalent term exists for the answer, provide it. Otherwise, leave the answer unchanged. Do not include any additional text or explanation."},
                #{"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nIf a one-word synonym or equivalent term exists for the answer, provide it. Otherwise, leave the answer unchanged."},
                #{"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nPlease provide a one-word synonym or an equivalent term for the answer."},
                #{"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nPlease rewrite the answer using synonyms or rephrasing."},
            ]     
            batch_prompts.append(messages)
    return batch_prompts

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["openchat", "llama-8b"],
        help="Model type to be fine-tuned."
    )
    parser.add_argument(
        "--root", type=str, default="/gaueko0/users/ietxarri010/GrAL_Irene/okvqa", help="Path to the OkVqa prediction files."
    )
    args = parser.parse_args()
    return args

def main():
    print("Parsing args...")
    args = parse_args()

    # Load jsons
    with open(os.path.join(args.root, 'train', f'annotations_train.json'), "r") as f: train = json.load(f)
    #with open(os.path.join(args.root, 'val', f'annotations_val.json'), "r") as f: val = json.load(f)

    # Create output files
    train_values = f"./output/values/train_values.txt"
    #val_values = f"./output/values/train_values.txt"

    os.makedirs(os.path.dirname(train_values), exist_ok=True)
    #os.makedirs(os.path.dirname(val_values), exist_ok=True)

    with open(train_values, 'w') as f: pass
    #with open(val_values, 'w') as f: pass
       

    if args.model_type == "llama-8b":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            pad_token_id=50256, #eos
        )

        batch_size = 8
        batch_prompts = get_llama_batch_prompts(train["annotations"][:10])

        for i in range(0, len(batch_prompts), batch_size):
            batch = batch_prompts[i:i+batch_size]
            outputs = pipeline(batch, max_new_tokens=10, truncation=True)

            for i, output in enumerate(outputs):
                #print(f"Equivalent answer {i}: {output['generated_text'][-1]}\n")
                print(output[0]['generated_text'][1]['content'])
                print(output[0]['generated_text'][-1]['content'])
                print("")
                
            #with open(train_values, 'a') as f:
            #    text = outputs[0]["generated_text"][-1]
            #    f.write(f'New answer: {text}\n\n')

        """
        for idx, ann in enumerate(train["annotations"][:5]):
            question = ann["question"]
            answers = [str(ans["answer"]) for ans in ann["answers"]]

            with open(train_values, 'a') as f:
                f.write(f'\nQ{idx}: {question}\n')

            for answer in answers:
                messages = [
                    {"role": "system", "content": "You are an assistant that provides rephrased answers using synonyms or equivalent phrases."},
                    {"role": "user", "content": f"Question: {question}\nAnswer: {answer}\nPlease rewrite the answer using synonyms or rephrasing."},
                ]
                outputs = pipeline(
                    messages,
                    max_new_tokens=256,
                )

                text = outputs[0]["generated_text"][-1]
                with open(train_values, 'a') as f:
                        f.write(f'Original answer: {answer}\n')
                        f.write(f'New answer: {text}\n\n')
        """
    
    elif args.model_type == "openchat":
        pipeline = transformers.pipeline("text-generation", model="openchat/openchat-3.5-0106", torch_dtype=torch.bfloat16, device_map="auto")

        for idx, ann in enumerate(train["annotations"][:5]):
            question = ann["question"]
            answers = [str(ans["answer"]) for ans in ann["answers"]]

            with open(train_values, 'a') as f:
                 f.write(f'\nQ{idx}: {question}\n')
            
            for answer in answers:
                #prompt = f"Question: {question}\nAnswer: {answer}\nPlease rewrite the answer using synonyms or rephrasing."
                #prompt = f"Question: {question}\nAnswer: {answer}\nPlease replace '{answer}' with a synonym or an equivalent term."
                prompt = f"Please replace '{answer}' with a synonym or an equivalent term."

                result = pipeline(prompt, max_length=20, truncation=True, num_return_sequences=1) 
                text = result[0]['generated_text']
                with open(train_values, 'a') as f:
                    f.write(f'Original answer: {answer}\n')
                    f.write(f'New answer: {text}\n\n')

if __name__ == "__main__":
    main()