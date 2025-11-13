# flake8: noqa

import sys
from importlib import resources


sys.path.insert(0, "/home/runai-home/Biomni/")


import argparse
import os

from biomni.agent import A1
from biomni.cogzero.canonicalizer import compare_outputs
from biomni.cogzero.prompt.proposer import PROPOSER_PROMPT
from biomni.cogzero.prompt.retriever import RETRIEVER_PROMPT
from biomni.cogzero.prompt.solver import SOLVER_PROMPT
from biomni.cogzero.utils import (
    clear_persistent_namespace,
    get_resources_from_agent,
    load_data,
    parse_solution_as_json,
    save_data,
)
from biomni.config import default_config


MODEL_DIR = "/mlbio_scratch/kodryan/hf_downloads/"
BIOMNI_DATA_DIR = "/mlbio_scratch/panigrah/agents/Biomni/data/"
DATA_DIR = "/mlbio_scratch/kodryan/Biomni/data/"


def parser():
    parser = argparse.ArgumentParser(description="Generate tabular reasoning tasks")
    parser.add_argument("--data_version", type=str, required=True, help="Data version")
    parser.add_argument("--trial", type=int, default=None, help="Trial number (None for the setup run)")
    parser.add_argument(
        "--model_id",
        type=str,
        default="gpt-oss-120b",
        help="Model ID that is hosted on the vLLM server",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=float,
        default=1500,
        help="Biomni timeout seconds parameter",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL of the vLLM server",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries per dataset",
    )
    parser.add_argument(
        "--max_validations",
        type=int,
        default=3,
        help="Maximum number of validation attempts per task",
    )
    # parser.add_argument(
    #     "--n_task_examples",
    #     type=int,
    #     default=3,
    #     help="The number of task examples to use in the prompt",
    # )

    args = parser.parse_args()
    return args


def main():
    args = parser()

    # Modify global defaults for consistency
    default_config.path = BIOMNI_DATA_DIR
    default_config.llm = os.path.join(MODEL_DIR, args.model_id)
    default_config.source = "Custom"
    default_config.base_url = args.base_url
    default_config.timeout_seconds = args.timeout_seconds

    # Initialize a dummy agent to pre-collect all available resources
    dummy_agent = A1()
    resources = get_resources_from_agent(dummy_agent)
    del dummy_agent

    datasets_path = os.path.join(DATA_DIR, "all_sc_datasets.json")
    datasets_dict = load_data(datasets_path, strict=True)

    tasks_data_dir = os.path.join(DATA_DIR, f"generated/tasks/{args.data_version}")
    tasks_data_path = os.path.join(tasks_data_dir, f"raw/trial_{args.trial}.json")
    all_tasks = load_data(tasks_data_path, default=[])

    # seed_tasks_path = os.path.join(tasks_data_dir, f"seed/data.json")
    # seed_tasks = load_data(seed_tasks_path)
    # if seed_tasks is not None:
    #     print(f"Successfully loaded seed tasks from {seed_tasks_path}")
    # else:
    #     print(f"Could not find seed tasks at {seed_tasks_path}")
    #     print("Will use default task examples")

    setup_run = args.trial is None

    if setup_run:
        configs_data = {
            "args": vars(args),
            "datasets_dict": datasets_dict,
            "retriever_prompt": RETRIEVER_PROMPT,
            "proposer_prompt": PROPOSER_PROMPT,
            "solver_prompt": SOLVER_PROMPT,
        }
        configs_path = os.path.join(tasks_data_dir, "configs.json")
        save_data(configs_data, configs_path, makedirs=True)

    if all_tasks and isinstance(all_tasks, list):
        seen_datasets = {task["dataset_path"] for task in all_tasks}
    else:
        seen_datasets = set()

    for dataset_path, dataset_description in datasets_dict.items():
        if dataset_path in seen_datasets:
            print(f"Seen dataset {dataset_path}, skipping...")
            continue
        else:
            print(f"Generating a new task for dataset {dataset_path}")
            seen_datasets.add(dataset_path)

        data_part_prompt = f"Path: {dataset_path}\nDescription: {dataset_description}\n\n"
        data_part_prompt = data_part_prompt.strip()

        # task_examples = None
        # if seed_tasks is not None:
        #     # Avoid sampling tasks from current table to alleviate duplicates generation
        #     task_examples = sample_tasks(seed_tasks, args.n_task_examples, except_tables=[table])
        # prompt = construct_gen_prompt(additional_authorized_imports, task_examples)

        if not setup_run:
            for t in range(args.max_retries):
                try:
                    retriever_agent = A1(resource_indices={})
                    retriever_format_dict = {
                        "data_path": data_part_prompt,
                        "tools": retriever_agent.retriever._format_resources_for_prompt(resources.get("tools", [])),
                        "data_lake": retriever_agent.retriever._format_resources_for_prompt(
                            resources.get("data_lake", [])
                        ),
                        "libraries": retriever_agent.retriever._format_resources_for_prompt(
                            resources.get("libraries", [])
                        ),
                    }
                    retriever_prompt = RETRIEVER_PROMPT.format(**retriever_format_dict).strip()
                    retriever_outputs = retriever_agent.go(retriever_prompt)
                    clear_persistent_namespace()

                    selected_indices = retriever_agent.retriever._parse_llm_response(retriever_outputs[1])

                    proposer_agent = A1(resource_indices=selected_indices, proposer_mode=True)
                    proposer_prompt = PROPOSER_PROMPT.format(data_path=data_part_prompt).strip()
                    proposer_outputs = proposer_agent.go(proposer_prompt)
                    clear_persistent_namespace()

                    task_dict = parse_solution_as_json(proposer_outputs[1])

                    assert "task_description" in task_dict, "No task description found!"
                    assert "ground_truth_answer" in task_dict, "No ground-truth answer found!"
                    # assert all(isinstance(v, str) for v in task_dict.values()), "Not all values are strings!"

                    task_validated = False
                    for _ in range(args.max_validations):
                        solver_agent = A1(resource_indices=selected_indices)
                        solver_format_dict = {
                            "task_description": task_dict["task_description"],
                            "data_path": data_part_prompt,
                        }
                        solver_prompt = SOLVER_PROMPT.format(**solver_format_dict).strip()
                        solver_outputs = solver_agent.go(solver_prompt)
                        clear_persistent_namespace()

                        solver_answer = parse_solution_as_json(solver_outputs[1])
                        # Handle cases where solver_answer might be a dict instead of string
                        if isinstance(solver_answer, dict):
                            # If dict, try to extract the answer field or use the whole dict as string
                            solver_answer_str = solver_answer.get("answer", str(solver_answer))
                        else:
                            solver_answer_str = solver_answer

                        # Ensure ground_truth_answer is also a string
                        gt_answer_str = task_dict["ground_truth_answer"]
                        if not isinstance(gt_answer_str, str):
                            gt_answer_str = str(gt_answer_str)

                        compare_results = compare_outputs(solver_answer_str, gt_answer_str)
                        if compare_results["equal"]:
                            task_validated = True
                            break

                    assert task_validated, "Task could not be validated!"

                    # If no task examples were provided, keep IDs as None
                    # task_example_ids = (
                    #     [task_example["task_id"] for task_example in task_examples]
                    #     if task_examples is not None
                    #     else None
                    # )

                    task_dict.update(
                        {
                            "dataset_path": dataset_path,
                            "dataset_description": dataset_description,
                            "resource_indices": selected_indices,
                            "proposer_log": proposer_outputs[0],
                        }
                    )

                    all_tasks.append(task_dict)
                    save_data(all_tasks, tasks_data_path, makedirs=True)
                    print(f"Saved {len(all_tasks)} data items to {tasks_data_path}")

                    break
                except Exception as e:
                    print(f"Attempt {t + 1} / {args.max_retries} was unsuccessful :(")
                    print(e)
                    if t == args.max_retries - 1:
                        print("All retries failed!")
                    else:
                        print("Trying again...")
                finally:
                    clear_persistent_namespace()


if __name__ == "__main__":
    main()
