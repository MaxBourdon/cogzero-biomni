# flake8: noqa

import sys


sys.path.insert(0, "/home/runai-home/Biomni/")


import argparse
import os

from biomni.agent import A1
from biomni.cogzero.canonicalizer import compare_outputs
from biomni.cogzero.prompt.solver import SOLVER_PROMPT
from biomni.cogzero.utils import (
    clear_persistent_namespace,
    load_data,
    load_tasks_as_dict,
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
        default="Qwen3-32B",
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
        help="Maximum number of retries per task",
    )

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

    tasks_data_path = os.path.join(DATA_DIR, f"generated/tasks/{args.data_version}/clean.json")
    tasks_dict = load_tasks_as_dict(tasks_data_path)
    task_ids = list(tasks_dict.keys())

    trajectories_data_dir = os.path.join(DATA_DIR, f"generated/trajectories/{args.data_version}/{args.model_id}")
    trajectories_data_path = os.path.join(trajectories_data_dir, f"trial_{args.trial}.json")
    all_trajectories = load_data(trajectories_data_path, default=[])

    setup_run = args.trial is None

    if setup_run:
        configs_data = {
            "args": vars(args),
            "tasks_data_path": tasks_data_path,
            "task_ids": task_ids,
            "solver_prompt": SOLVER_PROMPT,
        }
        configs_path = os.path.join(trajectories_data_dir, "configs.json")
        save_data(configs_data, configs_path, makedirs=True)

    if all_trajectories and isinstance(all_trajectories, list):
        seen_tasks = {trajectory["task_id"] for trajectory in all_trajectories}
    else:
        seen_tasks = set()

    for task_id in task_ids:
        if task_id in seen_tasks:
            print(f"Seen task {task_id}, skipping...")
            continue
        else:
            print(f"Generating a new trajectory for task {task_id}")
            seen_tasks.add(task_id)

        task = tasks_dict[task_id]

        necessary_fields = [
            "task_description",
            "ground_truth_answer",
            "dataset_path",
            "dataset_description",
            "resource_indices",
        ]
        missing_fields = [field for field in necessary_fields if field not in task]
        if missing_fields:
            print(f"Missing necessary fields {missing_fields} in task {task_id}, skipping...")
            continue

        task_description = task["task_description"]
        ground_truth_answer = task["ground_truth_answer"]
        dataset_path = task["dataset_path"]
        dataset_description = task["dataset_description"]
        resource_indices = task["resource_indices"]

        data_part_prompt = f"Path: {dataset_path}\nDescription: {dataset_description}\n\n"
        data_part_prompt = data_part_prompt.strip()

        if not setup_run:
            for t in range(args.max_retries):
                try:
                    solver_agent = A1(resource_indices=resource_indices)
                    solver_format_dict = {
                        "task_description": task_description,
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
                    gt_answer_str = ground_truth_answer
                    if not isinstance(gt_answer_str, str):
                        gt_answer_str = str(gt_answer_str)

                    compare_results = compare_outputs(solver_answer_str, gt_answer_str)
                    reward = float(compare_results["equal"])

                    memory_messages = solver_agent.get_full_trajectory()

                    print(f"Trajectory for task {task_id} is generated, got reward {reward}")

                    trajectory = {
                        "task_id": task_id,
                        "task_description": task_description,
                        "ground_truth_answer": ground_truth_answer,
                        "dataset_path": dataset_path,
                        "dataset_description": dataset_description,
                        "resource_indices": resource_indices,
                        "solver_answer": solver_answer,
                        "solver_answer_str": solver_answer_str,
                        "reward": reward,
                        "memory_messages": memory_messages,
                    }

                    all_trajectories.append(trajectory)
                    save_data(all_trajectories, trajectories_data_path, makedirs=True)
                    print(f"Saved {len(all_trajectories)} data items to {trajectories_data_path}")

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
