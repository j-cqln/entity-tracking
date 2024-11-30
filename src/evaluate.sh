# Examples

# t5-base, zero-shot, conversational phrasing, complete prompt style, implicit evaluation
python evaluate_t5.py --input_file="../data/conversational/complete_1-box-0-op.jsonl" --model_name="t5-base" --type="implicit"

# gpt2, few-shot with compact in-context demonstrations, not conversational phrasing, complete prompt style, explicit evaluation
python evaluate_gpt2.py --input_file="../data/not/complete_3-box-4-op.jsonl" --model_name="gpt2" --type="explicit" --few_shot --few_shot_type="compact"
