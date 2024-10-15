import transformers
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type", type=str, required=True, choices=["openchat", "llama-8b"],
        help="Model type to be fine-tuned."
    )

    args = parser.parse_args()
    return args

def main():
    print("Parsing args...")
    args = parse_args()

    if args.model_type == "llama-8b":
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": "Who are you?"},
        ]
        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )
        print(outputs[0]["generated_text"][-1])
    
    elif args.model_type == "openchat":
        pipeline = transformers.pipeline("text-generation", model="openchat/openchat-3.5-0106", torch_dtype=torch.bfloat16, device_map="auto")

        prompt = "¿When was America discovered?"

        result = pipeline(prompt, max_length=50, num_return_sequences=1) 

        print(result[0]['generated_text'], "\n")

if __name__ == "__main__":
    main()