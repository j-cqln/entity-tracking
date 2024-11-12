import argparse
import copy
import json
import itertools
import statistics

import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

SKIP_WORDS = ["", "and", ",", "the", "a", "an", "some", "absolutely", "both"]

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

def format_gold_responses(actual_content):
    if actual_content[0] != "nothing":
        gold_responses = []

        for permutation in itertools.permutations(actual_content):
            gold_response = ""

            for i, item in enumerate(permutation):
                gold_response += "the " + item

                if i == len(permutation) - 2:
                    if i == 0:
                        gold_response += " and "
                    else:
                        gold_response += ", and "
                elif i < len(permutation) - 2:
                    gold_response += ", "
        
            gold_responses.append(gold_response)
    else:
        gold_responses = copy.deepcopy(actual_content)
    
    return gold_responses

def generate_explicit(model, tokenizer, input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_new_tokens=20)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def generate_implicit(model, tokenizer, input_prompt):
    encoded_input = tokenizer(input_prompt, return_tensors="pt")
    logits = model(**encoded_input, labels=encoded_input["input_ids"]).logits
    return logits

def get_metrics_explicit(output, actual_content, initial_state, formerly_present, context):
    permutation_correct = 0
    errors = {
        "nothing_when_something": 0, "something_when_nothing": 0,
        "content_in_context": 0, "content_not_in_context": 0,
        "initial_state": 0, "formerly_present": 0
    }

    # Process output
    output = output.strip().strip(".")
    output_tokens = [tokenizer.decode(item, skip_special_tokens=True).strip() for item in tokenizer.encode(output)]

    # Correct if every non-skip word in output is in actual content and no others
    if (
        all(item in actual_content for item in output_tokens if item not in SKIP_WORDS) and
        all(item in output_tokens for item in actual_content)
        ):
        permutation_correct = 1
    else:
        permutation_correct = 0

        # Check every type of error
        # Should be nothing, but model predicted something
        if "nothing" in actual_content and "nothing" not in output_tokens:
            errors["nothing_when_something"] += 1

        # Should be something, but model predicted nothing
        if "nothing" not in actual_content and "nothing" in output_tokens:
            errors["something_when_nothing"] += 1

        # Wrong output but was in context (any from all content of all box)
        if any(item in context for item in output_tokens if item not in SKIP_WORDS and item not in actual_content):
            errors["content_in_context"] += 1

        # Wrong output was not in context
        if any(item not in context for item in output_tokens if item not in SKIP_WORDS and item not in actual_content):
            errors["content_not_in_context"] += 1
        
        # Wrong output was in initial state
        if any(item in initial_state for item in output_tokens if item not in SKIP_WORDS and item not in actual_content):
            errors["initial_state"] += 1
        
        # Wrong output was formerly present
        if any(item in formerly_present for item in output_tokens if item not in SKIP_WORDS and item not in actual_content):
            errors["formerly_present"] += 1
    
    return permutation_correct, errors

def get_metrics_implicit(outputs, gold_responses, start_index, actual_content, initial_state, formerly_present, context):
    decoded_top_outputs = []
    permutation_correct = []
    permutation_top_5_correct = []
    permutation_correct_ranks = []
    errors = []

    for output, gold_response in zip(outputs, gold_responses):
        # Outputs are logits; transform to probabilities
        output = torch.softmax(output, dim=-1)

        index = start_index

        top_output = []

        top_5_correct_bool = True
        correct_bool = True
        ranks = []
        error = {
            "nothing_when_something": 0, "something_when_nothing": 0,
            "content_in_context": 0, "content_not_in_context": 0,
            "initial_state": 0, "formerly_present": 0
        }

        for true_decoded_token in [tokenizer.decode(item, skip_special_tokens=True).strip() for item in tokenizer.encode(gold_response)]:
            if true_decoded_token not in SKIP_WORDS:
                # Get top 5 tokens by probability
                top_5 = tokenizer.decode(torch.topk(output.squeeze()[index], 5).indices, skip_special_tokens=True)

                # Top 5 accuracy
                if true_decoded_token not in top_5:
                    top_5_correct_bool = False

                # Get top 1 token by probability
                top_1 = tokenizer.decode(torch.argmax(output.squeeze()[index]), skip_special_tokens=True)
                top_output.append(top_1)

                # Correctness based on most probable token
                if true_decoded_token != top_1.strip():
                    correct_bool = False
                
                # Get rank of gold response
                sorted_tokens = torch.argsort(output.squeeze()[index], descending=True)
                sorted_decoded = [tokenizer.decode(sorted_token, skip_special_tokens=True) for sorted_token in sorted_tokens]
                sorted_decoded = [item.strip() for item in sorted_decoded]
                rank = sorted_decoded.index(true_decoded_token)
                ranks.append(rank)

            index += 1
        
        decoded_top_outputs.append(" ".join(top_output))
        
        # Metrics for this permutation
        if top_5_correct_bool:
            permutation_top_5_correct.append(1)
        else:
            permutation_top_5_correct.append(0)
        
        if correct_bool:
            permutation_correct.append(1)
        else:
            permutation_correct.append(0)
        
        rank = sum(ranks) / len(ranks)
        permutation_correct_ranks.append(rank)

        # Check every type of error
        # Should be nothing, but model predicted something
        if not correct_bool and "nothing" in actual_content and "nothing" not in top_output:
            error["nothing_when_something"] += 1

        # Should be something, but model predicted nothing
        if not correct_bool and "nothing" not in actual_content and "nothing" in top_output:
            error["something_when_nothing"] += 1

        # Wrong output but was in context (any from all content of all box)
        if not correct_bool and any(item in context for item in top_output if item not in SKIP_WORDS and item not in actual_content):
            error["content_in_context"] += 1

        # Wrong output was not in context
        if not correct_bool and any(item not in context for item in top_output if item not in SKIP_WORDS and item not in actual_content):
            error["content_not_in_context"] += 1
        
        # Wrong output was in initial state
        if not correct_bool and any(item in initial_state for item in top_output if item not in SKIP_WORDS and item not in actual_content):
            error["initial_state"] += 1
        
        # Wrong output was formerly present
        if not correct_bool and any(item in formerly_present for item in top_output if item not in SKIP_WORDS and item not in actual_content):
            error["formerly_present"] += 1
        
        errors.append(error)

    # Get argmin of permutation_correct_ranks
    argmin_ranks = permutation_correct_ranks.index(min(permutation_correct_ranks))
    
    return max(permutation_correct), max(permutation_top_5_correct), min(permutation_correct_ranks), errors[argmin_ranks], decoded_top_outputs[argmin_ranks]

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

    # Outputs (explicit: generated response, implicit: top 1 token)
    outputs = []

    # Metrics
    correct = 0
    top_5_correct = 0
    correct_ranks = []
    errors = {
        "nothing_when_something": 0, "something_when_nothing": 0,
        "content_in_context": 0, "content_not_in_context": 0,
        "initial_state": 0, "formerly_present": 0
    }

    # Evaluate model on each sample
    for sample in data:
        input_prompt = sample["input"]
        actual_content = sample["output"]
        initial_state = sample["initial_state"]
        formerly_present = sample["formerly_present"]
        context = sample["context"]

        # Generate correct output for each permutation of gold response
        gold_responses = format_gold_responses(actual_content)

        # Sample metrics for each permutation
        permutation_correct = []
        permutation_top_5_correct = []
        permutation_correct_ranks = []

        # Explicit
        if args.type == "explicit":
            # Additional processing based on model type
            if "t5" in args.model_name.lower():
                input_prompt += "<extra_id_0>"
            
            output = generate_explicit(model, tokenizer, input_prompt)
            outputs.append(output)
            
            # Metrics comparing output against all permutations
            permutation_correct, permutation_errors = get_metrics_explicit(
                output, actual_content, initial_state, formerly_present, context
            )
        
        # Implicit
        else:
            permutation_outputs = []

            # Length of input prompt
            start_index = len(tokenizer.encode(input_prompt)) - 1

            for gold_response in gold_responses:
                # Process input prompt for each permutation for teacher forcing
                processed_input_prompt = input_prompt
                processed_input_prompt += gold_response

                permutation_output = generate_implicit(model, tokenizer, processed_input_prompt)
                permutation_outputs.append(permutation_output)

            # Metrics from teaching forcing for all permutations
            permutation_correct, permutation_top_5_correct, permutation_correct_ranks, permutation_errors, top_1_output = get_metrics_implicit(
                permutation_outputs, gold_responses, start_index, actual_content, initial_state, formerly_present, context
            )

            outputs.append(top_1_output)

        # Metrics for the sample based on best permutation
        correct += permutation_correct

        if args.type == "implicit":
            top_5_correct += permutation_top_5_correct
            correct_ranks.append(permutation_correct_ranks)

        errors = {key: errors[key] + permutation_errors[key] for key in permutation_errors.keys()}

    accuracy = correct / len(data)

    if args.type == "implicit":
        top_5_accuracy = top_5_correct / len(data)
        correct_rank = statistics.median(correct_ranks) # Formerly (correct_ranks) / len(data)

    # Save outputs
    with open("{}_{}_{}_{}_outputs.txt".format(
        args.input_file.split("/")[2],
        args.input_file.split("/")[-1].strip(".jsonl"),
        args.model_name,
        args.type
    ), "w") as f:
        for output in outputs:
            f.write(f"{output}\n")

    # Calculate and save accuracy
    with open("{}_{}_{}_{}_accuracy.txt".format(
        args.input_file.split("/")[2],
        args.input_file.split("/")[-1].strip(".jsonl"),
        args.model_name,
        args.type
    ), "w") as f:
        f.write(f"accuracy\t{accuracy}\n")

        if args.type == "implicit":
            f.write(f"top_5_accuracy\t{top_5_accuracy}\n")
            f.write(f"mean_gold_response_rank\t{correct_rank}\n")
        
        for error in errors.keys():
            f.write(f"{error}\t{errors[error]}\n")
