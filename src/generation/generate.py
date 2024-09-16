import argparse
import pandas as pd

from utils import get_model_and_tokenizer, load_jsonl, predict

def main():
    parser = argparse.ArgumentParser("Command line utility for generating predictions.")
    parser.add_argument("--model_name", type=str,
                        help="Name of model.", required=True)
    parser.add_argument("--data_path", type=str,
                        help="Path to testing data in jsonl format.", required=True)
    parser.add_argument("--model_output", type=str,
                        help="Path to save model output in TSV format.", required=True)
    args = parser.parse_args()

    model_name = args.model_name
    data_path = args.data_path
    output_path = args.model_output

    model_kwargs = {
        "google/flan-t5-base": {"num_beams": 3, "num_return_sequences": 1, "max_length": 256},
        "gpt2": {"temperature": 0, "max_length": 150}
    }

    if model_name in model_kwargs:
        generation_kwargs = model_kwargs[model_name]
    else:
        generation_kwargs = None

    model, tokenizer = get_model_and_tokenizer(model_name)
    dataset = load_jsonl(data_path)

    results = []

    for item in dataset:
        pred, log_prob = predict(
            item["sentence_masked"],
            model,
            tokenizer,
            generation_kwargs=generation_kwargs,
            output_scores=True
        )

        results.append({
            "target": item["masked_content"],
            "prediction": pred,
            "input": item["sentence_masked"],
            "log_prob": log_prob
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df[["target", "prediction", "input"]] # Match format of existing work
    results_df.to_csv(output_path, sep="\t", index=False)

if __name__ == "__main__":
    main()