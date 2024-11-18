import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Open tab-separated accuracy files which have 2 columns, the first the type of statistic and the second the value
for conversational in ["conversational", "not"]:
    for prompt_type in ["complete", "question"]:
        for evaluation_type in ["explicit", "implicit"]:
            for model_name in ["t5-base"]:
                for few_shot in ["True", "False"]:
                    for boxes in [1, 3, 5]:
                        accuracy = []
                        top_5_accuracy = []
                        median_gold_response_rank = []

                        errors = {}

                        for ops in [0, 1, 2, 3, 4, 5]:
                            filename = f"{conversational}_{prompt_type}_{boxes}-box-{ops}-op_{model_name}_{evaluation_type}_few_shot_{few_shot}_metrics.txt"
                            
                            with open(filename) as f:
                                lines = f.readlines()
                                lines = [line.strip().split("\t") for line in lines]
                                accuracy.append(float(lines[0][1]))
                                top_5_accuracy.append(float(lines[1][1]))
                                median_gold_response_rank.append(float(lines[2][1]))

                                # Errors
                                if errors == {}:
                                    errors = {line[0]: [int(line[1])] for line in lines[3:]}
                                else:
                                    for line in lines[3:]:
                                        errors[line[0]].append(int(line[1]))
                            
                        # Accuracy line plot where x axis is number of operations
                        # Title is conversational, prompt type, evaluation type, and number of boxes
                        plt.ylim(0, 1)
                        plt.plot(range(6), accuracy)
                        plt.title(f"Accuracy, {conversational}, {prompt_type}, {boxes} boxes, {evaluation_type}")
                        plt.xlabel("Number of operations")
                        plt.ylabel("Accuracy")
                        plt.savefig(f"{conversational}_{prompt_type}_{boxes}_{evaluation_type}.png")
                        plt.close()

                        # Same plot for top 5 accuracy
                        plt.ylim(0, 1)
                        plt.plot(range(6), top_5_accuracy)
                        plt.title(f"Top 5 accuracy, {conversational}, {prompt_type}, {boxes} boxes, {evaluation_type}")
                        plt.xlabel("Number of operations")
                        plt.ylabel("Top 5 Accuracy")
                        plt.savefig(f"{conversational}_{prompt_type}_{boxes}_{evaluation_type}_top_5.png")
                        plt.close()

                        # Same plot for median gold response rank
                        plt.plot(range(6), median_gold_response_rank)
                        plt.title(f"Gold response rank, {conversational}, {prompt_type}, {boxes} boxes, {evaluation_type}")
                        plt.xlabel("Number of operations")
                        plt.ylabel("Median gold response rank")
                        plt.savefig(f"{conversational}_{prompt_type}_{boxes}_{evaluation_type}_median_gold_response_rank.png")
                        plt.close()

                        # Grouped bar graph for errors, horizontal axis is number of operations
                        # Title is conversational, prompt type, evaluation type, and number of boxes
                        # y axis is number of errors
                        x = np.arange(len(errors))  # the label locations
                        width = 0.1  # the width of the bars
                        multiplier = 0

                        fig, ax = plt.subplots(layout='constrained')

                        for attribute, measurement in errors.items():
                            offset = width * multiplier
                            rects = ax.bar(x + offset, measurement, width, label=attribute)
                            ax.bar_label(rects, padding=3)
                            multiplier += 1

                        # Add some text for labels, title and custom x-axis tick labels, etc.
                        ax.set_ylabel("Number of errors")
                        ax.set_title(f"Errors, {conversational}, {prompt_type}, {boxes} boxes, {evaluation_type}")
                        ax.set_xticks(x + width, range(6))
                        ax.legend(loc='upper left')
                        ax.set_ylim(0, 250)
                        plt.savefig(f"{conversational}_{prompt_type}_{boxes}_{evaluation_type}_errors.png")
                        plt.close()