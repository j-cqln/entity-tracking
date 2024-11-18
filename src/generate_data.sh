N_SAMPLES=1000

for boxes in 1 3 5
do
    for ops in {0..5}
    do
        MAX_ITEMS=$((ops+1))
        python generate_data.py --n_samples=$N_SAMPLES --output_path="../data/conversational/complete_${boxes}-box-${ops}-op.jsonl" --num_boxes=${boxes} --max_num_ops=${ops} --max_items_per_box=${MAX_ITEMS} --all_objects="../data/all_objects" --prompt_type="complete" --conversational
        python generate_data.py --n_samples=$N_SAMPLES --output_path="../data/conversational/question_${boxes}-box-${ops}-op.jsonl" --num_boxes=${boxes} --max_num_ops=${ops} --max_items_per_box=${MAX_ITEMS} --all_objects="../data/all_objects" --prompt_type="question" --conversational
        python generate_data.py --n_samples=$N_SAMPLES --output_path="../data/not/complete_${boxes}-box-${ops}-op.jsonl" --num_boxes=${boxes} --max_num_ops=${ops} --max_items_per_box=${MAX_ITEMS} --all_objects="../data/all_objects" --prompt_type="complete"
        python generate_data.py --n_samples=$N_SAMPLES --output_path="../data/not/question_${boxes}-box-${ops}-op.jsonl" --num_boxes=${boxes} --max_num_ops=${ops} --max_items_per_box=${MAX_ITEMS} --all_objects="../data/all_objects" --prompt_type="question"
    done
done