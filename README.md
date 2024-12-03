# entity-tracking
Prompting does not replace probability: implicit evaluation of entity tracking performance in language models

## Usage

Required packages can be found in `src/requirements.txt`. Python version >= 3.12.

### Data generation

Modify parameters accordingly `src/generate_data.sh` and run to generate data sets

### Evaluate

See `src/evaluate.sh` for examples of evaluating models on certain data sets.

### Visualize

Run `python visualize.py` in the same directory as the output files from the evaluation stage to generate figures visualizing results.
