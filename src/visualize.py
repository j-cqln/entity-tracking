import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Tab-separated metrics files, 2 columns (statistic type, statistic value)
# Individual plots
for model_name in ["t5-base", "gpt2"]:
    for few_shot in ["False_None", "True_compact", "True_full", "True_single"]:
        for prompt_type in ["complete"]:#, "question"]:
            for conversational in ["conversational", "not"]:
                for evaluation_type in ["explicit", "implicit"]:
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

                                if evaluation_type == "implicit":
                                    top_5_accuracy.append(float(lines[1][1]))
                                    median_gold_response_rank.append(float(lines[2][1]) + 1.0)

                                error_start_index = 3 if evaluation_type == "implicit" else 1

                                # Errors (proportion of all errors that are each error)
                                if errors == {}:
                                    if (1000 * (1 - accuracy[-1])) != 0:
                                        errors = {line[0]: [float(line[1]) / (1000 * (1 - accuracy[-1]))] for line in lines[error_start_index:]}
                                    else:
                                        errors = {line[0]: [0] for line in lines[error_start_index:]}
                                else:
                                    for line in lines[error_start_index:]:
                                        if (1000 * (1 - accuracy[-1])) != 0:
                                            errors[line[0]].append(float(line[1]) / (1000 * (1 - accuracy[-1])))
                                        else:
                                            errors[line[0]].append(0)
                            
                        # Accuracy line plot where x axis is number of operations
                        # Title is conversational, prompt type, evaluation type, and number of boxes
                        plt.ylim(-0.02, 1.02)
                        plt.plot(range(6), accuracy)
                        plt.title(f"Accuracy, {"conv." if conversational == "conversational" else "not"}, {boxes} box(es), {evaluation_type}, few-shot={few_shot.removesuffix("_None")}")
                        # plt.title(f"Accuracy, {"conv." if conversational == "conversational" else "not"}, {prompt_type}, {boxes} box(es), {evaluation_type}, few-shot={few_shot}")
                        plt.xlabel("Number of operations")
                        plt.ylabel("Accuracy")
                        plt.savefig(f"{model_name}_{conversational}_{prompt_type}_{boxes}_{evaluation_type}_few_shot_{few_shot}_accuracy.png")
                        plt.close()

                        if evaluation_type == "implicit":
                            # Same plot for top 5 accuracy
                            plt.ylim(-0.02, 1.02)
                            plt.plot(range(6), top_5_accuracy)
                            plt.title(f"Top 5 accuracy, {"conv." if conversational == "conversational" else "not"}, {boxes} box(es), {evaluation_type}, few-shot={few_shot.removesuffix("_None")}")
                            # plt.title(f"Top 5 accuracy, {"conv." if conversational == "conversational" else "not"}, {prompt_type}, {boxes} box(es), {evaluation_type}, few-shot={few_shot}")
                            plt.xlabel("Number of operations")
                            plt.ylabel("Top 5 Accuracy")
                            plt.savefig(f"{model_name}_{conversational}_{prompt_type}_{boxes}_{evaluation_type}_few_shot_{few_shot}_top_5.png")
                            plt.close()

                            # Same plot for median gold response rank
                            plt.plot(range(6), median_gold_response_rank)
                            plt.yscale("log")
                            plt.title(f"Gold response rank, {"conv." if conversational == "conversational" else "not"}, {boxes} box(es), {evaluation_type}, few-shot={few_shot.removesuffix("_None")}")
                            # plt.title(f"Gold response rank, {"conv." if conversational == "conversational" else "not"}, {prompt_type}, {boxes} box(es), {evaluation_type}, few-shot={few_shot}")
                            plt.xlabel("Number of operations")
                            plt.ylabel("Median gold response rank")
                            plt.savefig(f"{model_name}_{conversational}_{prompt_type}_{boxes}_{evaluation_type}_few_shot_{few_shot}_median_gold_response_rank.png")
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
                            # ax.bar_label(rects, padding=3)
                            multiplier += 1

                        # Add some text for labels, title and custom x-axis tick labels, etc.
                        ax.set_ylabel("Proportion of all errors")
                        ax.set_xlabel("Boxes (number of errors in parentheses)")
                        ax.set_title(f"Errors, {"conv." if conversational == "conversational" else "not"}, {boxes} box(es), {evaluation_type}, few-shot={few_shot.removesuffix("_None")}")
                        # ax.set_title(f"Errors, {"conv." if conversational == "conversational" else "not"}, {prompt_type}, {boxes} box(es), {evaluation_type}, few-shot={few_shot.removesuffix("_None")}")
                        ax.set_xticks(x + width, [f"{i} ({int((1 - accuracy[i]) * 1000)})" for i in range(6)])
                        ax.legend(loc='upper right', fontsize=5)
                        ax.set_ylim(0, 1.02)
                        plt.savefig(f"{model_name}_{conversational}_{prompt_type}_{boxes}_{evaluation_type}_few_shot_{few_shot}_errors.png")
                        plt.close()

# Combined plots, separate by few-shot, boxes
for few_shot in ["False_None", "True_compact", "True_full", "True_single"]:
    for prompt_type in ["complete"]:#, "question"]:
        for boxes in [1, 3, 5]:
            accuracies = []

            for model_name in ["t5-base", "gpt2"]:
                for evaluation_type in ["explicit", "implicit"]:
                    for conversational in ["conversational", "not"]:
                        accuracy = []

                        for ops in [0, 1, 2, 3, 4, 5]:
                            filename = f"{conversational}_{prompt_type}_{boxes}-box-{ops}-op_{model_name}_{evaluation_type}_few_shot_{few_shot}_metrics.txt"
                            
                            with open(filename) as f:
                                lines = f.readlines()
                                lines = [line.strip().split("\t") for line in lines]
                                accuracy.append(float(lines[0][1]))
                        
                        accuracies.append((accuracy, evaluation_type, conversational, model_name))

                plt.ylim(-0.02, 1.02)

                label_found = set()
                label_to_color = {
                    ("explicit", "t5-base"): "chartreuse",
                    ("implicit", "t5-base"): "dodgerblue",
                    ("explicit", "gpt2"): "mediumseagreen",
                    ("implicit", "gpt2"): "midnightblue",
                }

                for a in accuracies:
                    linestyle = "solid" if a[2] == "conversational" else "dashed"

                    if a[1] in label_found:
                        plt.plot(range(6), a[0], color=label_to_color[(a[1], a[3])], linestyle=linestyle)
                    else:
                        label_found.add(a[1])
                        plt.plot(range(6), a[0], color=label_to_color[(a[1], a[3])], label=a[1], linestyle=linestyle)

                plt.title(f"Accuracy, {boxes} box(es), few-shot={few_shot.removesuffix("_None")}")
                plt.xlabel("Number of operations")
                plt.ylabel("Accuracy")

                # plt.legend(loc="center right")

                plt.savefig(f"all_{conversational}_{boxes}_few_shot_{few_shot}.png")
                plt.close()

# Combined plots, separate by model, few-shot, boxes
for model_name in ["t5-base", "gpt2"]:
    for few_shot in ["False_None", "True_compact", "True_full", "True_single"]:
        for prompt_type in ["complete"]:#, "question"]:
            for boxes in [1, 3, 5]:
                accuracies = []

                for evaluation_type in ["explicit", "implicit"]:
                    for conversational in ["conversational", "not"]:
                        accuracy = []

                        for ops in [0, 1, 2, 3, 4, 5]:
                            filename = f"{conversational}_{prompt_type}_{boxes}-box-{ops}-op_{model_name}_{evaluation_type}_few_shot_{few_shot}_metrics.txt"
                            
                            with open(filename) as f:
                                lines = f.readlines()
                                lines = [line.strip().split("\t") for line in lines]
                                accuracy.append(float(lines[0][1]))
                        
                        accuracies.append((accuracy, evaluation_type, conversational))

                plt.ylim(-0.02, 1.02)

                label_found = set()
                label_to_color = {
                    "explicit": "chartreuse",
                    "implicit": "dodgerblue"
                }

                for a in accuracies:
                    linestyle = "solid" if a[2] == "conversational" else "dashed"

                    if a[1] in label_found:
                        plt.plot(range(6), a[0], color=label_to_color[a[1]], linestyle=linestyle)
                    else:
                        label_found.add(a[1])
                        plt.plot(range(6), a[0], color=label_to_color[a[1]], label=a[1], linestyle=linestyle)

                plt.title(f"Accuracy, {model_name}, {boxes} box(es), few-shot={few_shot.removesuffix("_None")}")
                plt.xlabel("Number of operations")
                plt.ylabel("Accuracy")

                # plt.legend(loc="center right")

                plt.savefig(f"all_{conversational}_{boxes}_few_shot_{few_shot}.png")
                plt.close()

# Data statistics
for filename, title in zip(
    ["unchanged", "nothing"],
    ["Proportional identical to start", "Proportion empty"]
    ):

    with open(f"{filename}_proportions.out") as f:
        lines = f.readlines()
        lines = [float(line.strip()) for line in lines]
        index = 0

        for boxes in [1, 3, 5]:
            conversational_complete = []
            conversational_question = []
            not_complete = []
            not_question = []

            for ops in [0, 1, 2, 3, 4, 5]:
                conversational_complete.append(lines[index])
                conversational_question.append(lines[index + 1])
                not_complete.append(lines[index + 2])
                not_question.append(lines[index + 3])
                index += 4

            for lst, typ in zip(
                [conversational_complete, conversational_question, not_complete, not_question],
                ["complete", "question", "complete", "question"]
                ):
                plt.ylim(-0.02, 1.02)
                plt.plot(range(6), lst)
                plt.title(f"{title}, {typ}, {boxes} boxes")
                plt.xlabel("Number of operations")
                plt.ylabel("Proportion")
                plt.savefig(f"{filename}_{typ}_{boxes}.png")
                plt.close()
