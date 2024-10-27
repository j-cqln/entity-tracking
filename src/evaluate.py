import argparse
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

def load_jsonl(data_path):
    with open(data_path, "r") as file:
        data = [json.loads(line) for line in file]

    return data

def get_model_and_tokenizer(model_name):
    if "t5" in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def generate_explicit(model, tokenizer, input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_new_tokens=20)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def generate_implicit(model, tokenizer, input_prompt):
    encoded_input = tokenizer(input_prompt, return_tensors="pt")
    logits = model(**encoded_input, labels=encoded_input["input_ids"]).logits
    return logits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses and evaluate accuracy")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input file")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--type", type=str, required=True, help="Type of generation task.")

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(args.model_name)
    
    # Read input file
    data = load_jsonl(args.input_file)

    outputs = []

    correct = 0
    top_5_correct = 0
    correct_ranks = []

    # Generate responses and evaluate correctness
    for item in data:
        input_prompt = item["input"]
        gold_response = item["output"]

        if "t5" in args.model_name.lower():
            input_prompt += "<extra_id_0>"

        # Explicit
        if args.type == "explicit":
            output = generate_explicit(model, tokenizer, input_prompt)
            output = output.strip()
            outputs.append(output)

            # Correctness
            if output.strip(".") == gold_response.strip("."):
                correct += 1
        
        # Implicit
        else:
            # Length of input prompt
            input_length = len(tokenizer.encode(input_prompt)) - 1

            # Append gold response for teacher forcing
            input_prompt += gold_response
            
            gold_response = gold_response.strip(".")

            output = generate_implicit(model, tokenizer, input_prompt)

            # Outputs are logits
            # Transform to probabilities
            output = torch.softmax(output, dim=-1)

            index = input_length

            top_output = []

            top_5_correct_bool = True
            correct_bool = True
            ranks = []

            for token in [tokenizer.decode(item, skip_special_tokens=True).strip() for item in tokenizer.encode(gold_response)]:
                if token != "":
                    # Get top 5 tokens by probability
                    top_5 = tokenizer.decode(torch.topk(output.squeeze()[index], 5).indices, skip_special_tokens=True)

                    # Top 5 accuracy
                    if token not in top_5:
                        top_5_correct_bool = False

                    # Get top 1 token by probability
                    top_1 = tokenizer.decode(torch.argmax(output.squeeze()[index]), skip_special_tokens=True)
                    top_output.append(top_1)

                    # Correctness based on most probable token
                    if token != top_1.strip():
                        correct_bool = False
                    
                    # Get rank of gold response
                    sorted_tokens = torch.argsort(output.squeeze()[index], descending=True)
                    sorted_decoded = [tokenizer.decode(sorted_token, skip_special_tokens=True) for sorted_token in sorted_tokens]
                    sorted_decoded = [item.strip() for item in sorted_decoded]
                    rank = sorted_decoded.index(token)
                    ranks.append(rank)

                    index += 1
            
            outputs.append(" ".join(top_output))
            
            if top_5_correct_bool:
                top_5_correct += 1
            
            if correct_bool:
                correct += 1
            
            rank = sum(ranks) / len(ranks)
            correct_ranks.append(rank)
    
    accuracy = correct / len(data)

    if args.type == "implicit":
        top_5_accuracy = top_5_correct / len(data)
        correct_rank = sum(correct_ranks) / len(data)

    # Save outputs
    with open("{}_{}_{}_{}_outputs.txt".format(args.input_file.split("/")[-1].strip(".jsonl"), args.input_file.split("/")[2], args.model_name, args.type), "w") as f:
        for output in outputs:
            f.write(f"{output}\n")

    # Calculate and save accuracy
    with open("{}_{}_{}_{}_accuracy.txt".format(args.input_file.split("/")[-1].strip(".jsonl"), args.input_file.split("/")[2], args.model_name, args.type), "w") as f:
        f.write(f"Accuracy: {accuracy}\n")

        if args.type == "implicit":
            f.write(f"Top 5 accuracy: {top_5_accuracy}\n")
            f.write(f"Mean rank of gold response: {correct_rank}\n")
