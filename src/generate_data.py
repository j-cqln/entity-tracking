import argparse
import random
import json

INITIALIZATION = {
    "item": "Box {box} contains the {content}. ",
    "empty": "Box {box} is empty. "
}

OPERATIONS = {
    "put": "Put the {content} in Box {box}. ",
    "move": "Move the {content} from Box {from_box} to Box {to_box}. ",
    "remove": "Remove the {content} from Box {box}. "
}

OPERATIONS_CONVERSATIONAL = {
    "put": "I put the {content} in Box {box}. ",
    "move": "I move the {content} from Box {from_box} to Box {to_box}. ",
    "remove": "I remove the {content} from Box {box}. "
}

PROMPTS = {
    "question": "What is in Box {box}? ",
    "complete": "Box {box} contains "
}

PROMPTS_CONVERSATIONAL = {
    "question": "What will you find if you open Box {box}? ",
    "complete": "If you open Box {box}, you will find "
}

# VERBS = {
#     "put": ["Put", "Place"],
#     "move": ["Move", "Transfer"]
# }

# Few-shot prompt
# TBD

class WorldState:
    def __init__(self, num_boxes, max_num_ops, max_items_per_box, all_objects, conversational=False):
        self._boxes = [set() for _ in range(num_boxes)]
        self._num_boxes = num_boxes
        self._max_num_ops = max_num_ops
        self._max_items_per_box = max_items_per_box
        self._all_objects = all_objects.copy()

        self._input_data = ""

        if conversational:
            self._initialization = INITIALIZATION
            self._operations = OPERATIONS_CONVERSATIONAL
            self._prompts = PROMPTS_CONVERSATIONAL
        else:
            self._initialization = INITIALIZATION
            self._operations = OPERATIONS
            self._prompts = PROMPTS

        self._initialize()
        self._generate()

    def _initialize(self):
        for box in range(self._num_boxes):
            init = random.choice(list(self._initialization.keys()))

            if init == "empty":
                self._input_data += self._initialization["empty"].format(box=box)
            elif init == "item":
                content = self._all_objects.pop(random.randrange(len(self._all_objects)))
                self._boxes[box].add(content)
                self._input_data += self._initialization["item"].format(box=box, content=content)

    def _generate(self):
        # Start with empty boxes
        # Place 0 or 1 objects in each box
        # Perform operations
        # For each operation, we are either introducing a new object or moving an existing object
        for _ in range(self._max_num_ops):
            op = random.choice(list(self._operations.keys()))
            box = random.randint(0, self._num_boxes - 1)

            if op == "put":
                content = self._all_objects.pop(random.randrange(len(self._all_objects)))
                successful = self._put(content, box)
                
            elif op == "move":
                box_2 = random.randint(0, self._num_boxes - 1)
                successful = self._move(box, box_2)
            
            elif op == "remove":
                if len(self._boxes[box]) > 0:
                    content = random.choice(list(self._boxes[box]))
                    successful = self._remove(content, box)
                else:
                    successful = False

            if not successful:
                print("failed op")

    def _put(self, content, box):
        if len(self._boxes[box]) < self._max_items_per_box:
            self._boxes[box].add(content)
            self._input_data += self._operations["put"].format(content=content, box=box)
            return True
        else:
            return False
        
    def _move(self, from_box, to_box):
        if len(self._boxes[from_box]) > 0 and len(self._boxes[to_box]) < self._max_items_per_box:
            content = random.choice(list(self._boxes[from_box]))
            self._boxes[from_box].remove(content)
            self._boxes[to_box].add(content)
            self._input_data += self._operations["move"].format(content=content, from_box=from_box, to_box=to_box)
            return True
        else:
            return False
    
    def _remove(self, content, box):
        if content in self._boxes[box]:
            self._boxes[box].remove(content)
            self._input_data += self._operations["remove"].format(content=content, box=box)
            return True
        else:
            return False
    
    def _format_output(self, box):
        if len(self._boxes[box]) == 0:
            return "nothing."

        output = ""

        for i, item in enumerate(self._boxes[box]):
            output += "the " + item

            if i == len(self._boxes[box]) - 2:
                if i == 0:
                    output += " and "
                else:
                    output += ", and "
            elif i < len(self._boxes[box]) - 2:
                output += ", "
        
        output += "."
        
        return output

    def get_input_output_pair(self, box, prompt_type):
        if box < 0 or box >= self._num_boxes:
            raise ValueError(f"Invalid box number {box}.")
        else:
            return [self._input_data + self._prompts[prompt_type].format(box=box), self._format_output(box)]

if __name__ == "__main__":
    # Get all_objects path from argparse
    parser = argparse.ArgumentParser(description="Generate data for evaluating entity tracking")

    parser.add_argument("--n_samples", type=int, required=True, help="Number of samples to generate")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the generated data")

    parser.add_argument("--num_boxes", type=int, required=True, help="Number of boxes")
    parser.add_argument("--max_num_ops", type=int, required=True, help="Maximum number of operations")
    parser.add_argument("--max_items_per_box", type=int, required=True, help="Maximum number of items per box")
    parser.add_argument("--all_objects", type=str, required=True, help="Path to file containing all objects")
    parser.add_argument("--conversational", action="store_true", help="Use conversational prompts")
    parser.add_argument("--prompt_type", type=str, default="question", help="Prompt type (question/complete)")
    
    args = parser.parse_args()

    with open(args.all_objects, "r") as file:
        all_objects = file.read().splitlines()
    
    data = []

    while len(data) < args.n_samples:
        world_state = WorldState(args.num_boxes, args.max_num_ops, args.max_items_per_box, all_objects, args.conversational)

        # Store input-output pair for each box in each sample
        for box in range(args.num_boxes):
            data.append(world_state.get_input_output_pair(box, args.prompt_type))

            if len(data) == args.n_samples:
                break
    
    # Save as jsonl
    with open(args.output_path, "w") as file:
        for item in data:
            file.write(json.dumps({"input": item[0], "output": item[1]}) + "\n")