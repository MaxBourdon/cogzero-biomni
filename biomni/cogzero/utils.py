import gc
import glob
import json
import os
import re
from typing import Any, List, Optional

import numpy as np


def save_data(data: Any, path: str, makedirs: bool = False):
    "Save data to path as a JSON file."
    if makedirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"Failed to save data to {path}: {e}") from e


def load_data(path: str, default: Any = None, strict: bool = False) -> Any:
    "Load a JSON file from path; if not strict, return default in case of failure."
    data = default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        if strict:
            raise RuntimeError(f"Failed to load data from {path}: {e}") from e
        else:
            print(f"Warning: Failed to load data from {path}: {e}")
            print(f"Returning default: {default}")
    return data


def create_next_trial_dir(path: str, setup_run: bool = False) -> str:
    """
    Scans the given directory `path` for subdirectories with names matching the pattern 'trial_<number>'.
    Finds the highest <number>, creates a new directory named 'trial_<number+1>' (or 'trial_0' if none exists)
    if `setup_run` is False, and returns the new directory path.
    """
    os.makedirs(path, exist_ok=True)

    max_n = -1
    pattern = re.compile(r"^trial_(\d+)$")
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            m = pattern.match(name)
            if m:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n

    next_n = max_n + 1
    new_dir = os.path.join(path, f"trial_{next_n}")
    if not setup_run:
        os.makedirs(new_dir, exist_ok=False)
    return new_dir


def make_var_name(s: str) -> str:
    """
    Convert arbitrary string to a valid Python identifier:
    - Lowercase
    - Replace non-alphanumeric characters with underscores
    - Collapse multiple underscores
    - Strip leading/trailing underscores
    - Ensure it starts with letter or underscore
    """
    # Lowercase
    s = s.lower()
    # Replace non-alphanumeric with underscores
    s = re.sub(r"[^0-9a-z]+", "_", s)
    # Collapse multiple underscores
    s = re.sub(r"__+", "_", s)
    # Strip leading/trailing underscores
    s = s.strip("_")
    # Ensure it starts with a letter or underscore
    if not re.match(r"[a-z_]", s):
        s = "_" + s
    return s


def normalize_answer(text: Optional[str]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip()).lower()


def get_file_name(path_to_file: Optional[str]) -> str:
    """Get the filename from a path, handling None and invalid paths."""
    if path_to_file is None:
        return ""
    return os.path.basename(path_to_file)


def sample_tasks(seed_tasks: List[dict], n_samples: int, except_tables: Optional[List[str]] = None) -> List[dict]:
    """
    Sample tasks from seed_tasks, optionally excluding tasks from certain tables.

    Args:
        seed_tasks: List of task dictionaries
        n_samples: Number of samples to return
        except_tables: Optional list of table names to exclude

    Returns:
        List of sampled task dictionaries

    Raises:
        ValueError: If n_samples is invalid or no tasks available
    """
    if not seed_tasks:
        return []

    if n_samples <= 0:
        return []

    if except_tables is not None:
        # Filter out tasks with certain tables
        filtered_tasks = [
            task for task in seed_tasks if get_file_name(task.get("path_to_table", "")) not in except_tables
        ]
    else:
        filtered_tasks = seed_tasks

    if not filtered_tasks:
        return []

    # Ensure we don't try to sample more than available
    n_samples = min(n_samples, len(filtered_tasks))

    # Check for the 'score' key in tasks
    if all("score" in task for task in filtered_tasks):
        scores = [task.get("score", 0.0) for task in filtered_tasks]
    else:
        scores = [1.0] * len(filtered_tasks)

    scores = np.array(scores)
    total_score = sum(scores)

    # Handle division by zero
    if total_score == 0:
        # If all scores are zero, use uniform distribution
        probs = np.ones(len(filtered_tasks)) / len(filtered_tasks)
    else:
        probs = scores / total_score

    # Sample tasks
    sampled_indices = np.random.choice(len(filtered_tasks), n_samples, replace=False, p=probs)
    sampled_tasks = [filtered_tasks[i] for i in sampled_indices]

    return sampled_tasks


def load_tasks_as_dict(tasks_path: str):
    tasks = load_data(tasks_path, strict=True)
    tasks = {task["task_id"]: task for task in tasks}
    return tasks


def parse_solution_as_json(last_message):
    execute_match = re.search(r"<solution>(.*?)</solution>", last_message, re.DOTALL)
    if execute_match is None:
        raise ValueError("No <solution> tag found in the last message")
    solution = execute_match.group(1)
    try:
        solution = json.loads(solution)
    except json.JSONDecodeError:
        print("Could not parse JSON, returning a raw solution string.")

    return solution


def get_resources_from_agent(agent):
    # Gather all available resources
    # 1. Tools from the registry
    all_tools = agent.tool_registry.tools

    # 2. Data lake items with descriptions
    data_lake_path = agent.path + "/data_lake"
    data_lake_content = glob.glob(data_lake_path + "/*")
    data_lake_items = [x.split("/")[-1] for x in data_lake_content]

    # Create data lake descriptions for retrieval
    data_lake_descriptions = []
    for item in data_lake_items:
        description = agent.data_lake_dict.get(item, f"Data lake item: {item}")
        data_lake_descriptions.append({"name": item, "description": description})

    # 3. Libraries with descriptions - use library_content_dict directly
    library_descriptions = []
    for lib_name, lib_desc in agent.library_content_dict.items():
        library_descriptions.append({"name": lib_name, "description": lib_desc})

    resources = {
        "tools": all_tools,
        "data_lake": data_lake_descriptions,
        "libraries": library_descriptions,
    }

    return resources


def clear_persistent_namespace():
    # Access the persistent namespace used by run_python_repl
    from biomni.tool.support_tools import _persistent_namespace

    # Clear the execution namespace
    if _persistent_namespace:
        _persistent_namespace.clear()
        gc.collect()
