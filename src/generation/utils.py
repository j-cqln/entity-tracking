import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, Trainer, TrainingArguments

def get_model_and_tokenizer(model_name):
    if "t5" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif "gpt" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise NotImplementedError("Only implemented for T5 and GPT models.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_jsonl(data_path):
    with open(data_path, "r") as file:
        data = [json.loads(line) for line in file]
    return data

def predict_t5(inputs, model, tokenizer, generation_kwargs=None, output_scores=False):
    input_ids = tokenizer(inputs, return_tensors="pt").input_ids

    if generation_kwargs:
        outputs = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=output_scores,
            **generation_kwargs
        )
    else:
        outputs = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=output_scores
        )

    prediction = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0]

    if output_scores:
        # Beam
        if "num_beams" in generation_kwargs:
            transition_scores = model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                outputs.beam_indices,
                normalize_logits=False
            )
            log_prob = transition_scores.sum().item()
        
        # Greedy
        else:
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            log_prob = transition_scores[0][:-1].sum().item()

        return prediction, log_prob

    return prediction

def predict_gpt(inputs, model, tokenizer, generation_kwargs=None, output_scores=False):
    pass

def predict(inputs, model, tokenizer, generation_kwargs=None, output_scores=False):
    if "t5" in model.config.model_type:
        return predict_t5(inputs, model, tokenizer, generation_kwargs, output_scores)
    elif "gpt" in model.config.model_type:
        return predict_gpt(inputs, model, tokenizer, generation_kwargs, output_scores)
    else:
        raise NotImplementedError("Only implemented for T5 and GPT models.")

def finetune():
    pass