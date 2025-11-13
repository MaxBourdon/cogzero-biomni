# A bunch of utils for the notebooks

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Union, cast

import matplotlib.pyplot as plt
import numpy as np

from .utils import load_data


# import seaborn as sns
# from matplotlib_venn import venn2


Trajectory = Mapping[str, Any]
TrajectoriesByTrial = Mapping[str, Sequence[Trajectory]]
StatsMapping = Mapping[str, Mapping[Any, float]]
TaskId = Union[int, str]


def load_data_dict(path_to_dir: Path | str) -> dict[str, list[Any]]:
    directory = Path(path_to_dir)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    trials: list[tuple[str, int]] = []
    for child in directory.iterdir():
        if not child.is_file():
            continue
        prefix, _, suffix = child.name.partition("_")
        suffix, dot, ext = suffix.partition(".")
        if prefix != "trial" or ext != "json" or not suffix.isdigit():
            continue
        trial_name = f"{prefix}_{suffix}"
        trials.append((trial_name, int(suffix)))

    data_dict: dict[str, list[Any]] = {}
    for trial_name, _ in sorted(trials, key=lambda item: item[1]):
        data_file = directory / f"{trial_name}.json"
        if not data_file.is_file():
            raise FileNotFoundError(f"Missing data file at {data_file}")
        loaded_data = cast(list[Any], load_data(data_file))
        data_dict[trial_name] = loaded_data

    return data_dict


def get_trajectories_subset(
    traj_dict: TrajectoriesByTrial,
    subset_values: Collection[Any],
    key: str = "dataset_name",
) -> TrajectoriesByTrial:
    """Return a subset of trajectories with ``traj[key]`` in ``subset_values``."""

    subset_lookup = set(subset_values)
    subtraj_dict: TrajectoriesByTrial = {}
    for trial, trajectories in traj_dict.items():
        subtraj_dict[trial] = [traj for traj in trajectories if traj[key] in subset_lookup]
    return subtraj_dict


def get_traj_by_datasets(
    traj_dict: TrajectoriesByTrial, datasets: Collection[str] | None = None
) -> dict[str, list[Trajectory]]:
    grouped: dict[str, list[Trajectory]] = {ds: [] for ds in datasets} if datasets is not None else {}

    for trajectories in traj_dict.values():
        for traj in trajectories:
            ds = str(traj["dataset_name"])
            grouped.setdefault(ds, []).append(traj)

    return grouped


def get_traj_by_tasks(
    traj_dict: TrajectoriesByTrial, tasks: dict[TaskId, Any] | None = None
) -> dict[TaskId, list[Trajectory]]:
    grouped: dict[TaskId, list[Trajectory]]
    if tasks is None:
        grouped = {}
    else:
        grouped = {task_id: [] for task_id in tasks.keys()}

    for trajectories in traj_dict.values():
        for traj in trajectories:
            task_id = traj["task_id"]
            grouped.setdefault(task_id, []).append(traj)

    return grouped


def get_stats(
    traj_by_items: Mapping[Any, Sequence[Trajectory]],
) -> StatsMapping:
    stats_dict: StatsMapping = {
        "counts": {item: 0.0 for item in traj_by_items},
        "rewards": {item: 0.0 for item in traj_by_items},
        "n_messages": {item: 0.0 for item in traj_by_items},
        "n_steps": {item: 0.0 for item in traj_by_items},
    }

    is_agent_step = (
        lambda message: isinstance(message, dict)
        and message.get("type") == "ai"
        and not message.get("content", "").startswith("<observation>")
    )

    for item, trajectories in traj_by_items.items():
        for traj in trajectories:
            reward = float(traj["reward"])
            stats_dict["counts"][item] += 1.0
            stats_dict["rewards"][item] += reward
            message_key = "memory_messages" if "memory_messages" in traj else "messages"
            messages = traj.get(message_key)
            stats_dict["n_messages"][item] += float(len(messages))
            stats_dict["n_steps"][item] += sum(1.0 for message in messages if is_agent_step(message))

    return stats_dict


def print_ratio(title: str, n_suc: float | int, n_tot: float | int, indent: int = 0) -> None:
    percentage = 100 * n_suc / n_tot if n_tot > 0 else 0.0
    print(f"{title:{indent}}\t{n_suc} of {n_tot} ({percentage:.2f}%)")


def print_statistics(arr: Iterable[float] | np.ndarray, indent: int = 5) -> None:
    if isinstance(arr, np.ndarray):
        data = arr.astype(float, copy=False)
    else:
        data = np.asarray(list(arr), dtype=float)

    if data.size == 0:
        print("No data available")
        return

    print(
        f"Count: {data.size:{indent}}, "
        f"Mean: {data.mean():{indent}.2f}, "
        f"Std: {data.std(ddof=1):{indent}.2f}, "
        f"Min: {data.min():{indent}.2f}, "
        f"Median: {np.median(data):{indent}.2f}, "
        f"Max: {data.max():{indent}.2f}"
    )


def print_by_trials(traj_dict: TrajectoriesByTrial) -> None:
    tot_suc = tot_cnt = 0
    for trial, trajectories in traj_dict.items():
        rewards = [traj["reward"] for traj in trajectories]
        n_suc = sum(rewards)
        n_tot = len(rewards)
        tot_suc += n_suc
        tot_cnt += n_tot
        print_ratio(trial, n_suc, n_tot, indent=8)
    print(40 * "-")
    print_ratio("Total", tot_suc, tot_cnt, indent=8)


def print_by_datasets(ds_stats: StatsMapping, datasets: Sequence[str] | None = None) -> None:
    if datasets is None:
        datasets = list(ds_stats["rewards"].keys())

    max_ds_len = max(len(ds) for ds in datasets)

    for ds in datasets:
        n_suc = ds_stats["rewards"]
        n_tot = ds_stats["counts"]
        print_ratio(ds, n_suc[ds], n_tot[ds], indent=max_ds_len)


def print_task_stat(
    task_stats: StatsMapping,
    task_ids: Sequence[TaskId] | None = None,
) -> None:
    if task_ids is None:
        task_ids = list(task_stats["rewards"].keys())

    rewards = np.array([task_stats["rewards"][t] for t in task_ids])
    counts = np.array([task_stats["counts"][t] for t in task_ids])
    n_steps = np.array([task_stats["n_steps"][t] for t in task_ids]) / counts
    suc_mask = rewards > 0

    print("Rewards ", end="\t")
    print_statistics(rewards)
    print("Rewards+", end="\t")
    print_statistics(rewards[suc_mask])
    print("Steps   ", end="\t")
    print_statistics(n_steps)
    print("Steps+  ", end="\t")
    print_statistics(n_steps[suc_mask])


def print_task_comp_stat(
    task_stats_before: StatsMapping,
    task_stats_after: StatsMapping,
    task_ids: Sequence[TaskId] | None = None,
) -> None:
    if task_ids is None:
        task_ids = list(task_stats_before["rewards"].keys())

    print("BEFORE")
    print_task_stat(task_stats_before, task_ids)
    print()
    print("AFTER")
    print_task_stat(task_stats_after, task_ids)
    print(40 * "-")

    rewards_before = np.array([task_stats_before["rewards"][t] for t in task_ids])
    rewards_after = np.array([task_stats_after["rewards"][t] for t in task_ids])

    suc_mask_before = rewards_before > 0
    suc_mask_after = rewards_after > 0

    n_tot = len(task_ids)
    indent = 10

    n_bga = sum(rewards_before > rewards_after)
    n_bla = sum(rewards_before < rewards_after)
    n_bea = sum(rewards_before == rewards_after)

    print_ratio("B > A", n_bga, n_tot, indent)
    print_ratio("B < A", n_bla, n_tot, indent)
    print_ratio("B = A", n_bea, n_tot, indent)

    print(40 * "-")

    n_sb = sum(suc_mask_before)
    n_sa = sum(suc_mask_after)

    print_ratio("B+", n_sb, n_tot, indent)
    print_ratio("A+", n_sa, n_tot, indent)

    print(40 * "-")

    n_sb_sa = sum(suc_mask_before & suc_mask_after)
    n_sb_fa = sum(suc_mask_before & ~suc_mask_after)
    n_fb_sa = sum(~suc_mask_before & suc_mask_after)
    n_fb_fa = sum(~suc_mask_before & ~suc_mask_after)

    print_ratio("B+ & A+", n_sb_sa, n_tot, indent)
    print_ratio("B+ & A-", n_sb_fa, n_tot, indent)
    print_ratio("B- & A+", n_fb_sa, n_tot, indent)
    print_ratio("B- & A-", n_fb_fa, n_tot, indent)


def plot_ds_cr(
    ds_stats: StatsMapping,
    datasets: Sequence[str] | None = None,
    mark_datasets: Collection[str] | None = None,
) -> None:
    if datasets is None:
        datasets = list(ds_stats["rewards"].keys())

    # Ensure both dicts have the same dataset names
    rewards = [ds_stats["rewards"][t] for t in datasets]
    counts = [ds_stats["counts"][t] for t in datasets]

    x = np.arange(len(datasets))  # x-axis positions
    width = 0.4  # bar width

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot counts (bottom histogram)
    ax1.bar(x, counts, width, label="Counts", color="lightblue")

    # Plot rewards (overlay histogram, narrower width)
    ax1.bar(x, rewards, width * 0.5, label="Rewards", color="orange")

    if mark_datasets is not None:
        # Mark specific datasets
        mark_ids = [datasets.index(ds) for ds in mark_datasets if ds in datasets]
        x_mark = x[mark_ids]
        ax1.plot(x_mark, np.zeros_like(x_mark), "k^")

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha="right")
    ax1.set_ylabel("Values")
    ax1.set_title("Counts and Rewards per Dataset")
    ax1.legend()

    plt.tight_layout()
    plt.show()


def plot_ds_comp(
    ds_stats_dicts: Sequence[StatsMapping],
    datasets: Sequence[str] | None = None,
    mark_datasets: Collection[str] | None = None,
    labels: Sequence[str] | None = ("Before", "After"),
) -> None:
    if datasets is None:
        datasets = list(ds_stats_dicts[0]["rewards"].keys())

    suc_rates_list = [
        [(ds_stats["rewards"][t] / ds_stats["counts"][t] if ds_stats["counts"][t] > 0 else 0) for t in datasets]
        for ds_stats in ds_stats_dicts
    ]

    x = np.arange(len(datasets))  # x-axis positions
    width = 0.4  # bar width

    fig, ax1 = plt.subplots(figsize=(8, 5))

    for i, suc_rate in enumerate(suc_rates_list):
        label = None if labels is None else labels[i]
        ax1.bar(x, suc_rate, width, label=label, alpha=0.75)
        width /= 2

    if mark_datasets is not None:
        # Mark specific datasets
        mark_ids = [datasets.index(ds) for ds in mark_datasets if ds in datasets]
        x_mark = x[mark_ids]
        ax1.plot(x_mark, np.zeros_like(x_mark), "k^")

    # X-axis labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha="right")
    ax1.set_ylabel("Success Rate")
    ax1.set_title("Success Rate per Dataset")
    ax1.legend()

    plt.tight_layout()
    plt.show()


def plot_task_comp(
    task_stats_dicts: Sequence[StatsMapping],
    task_ids: Sequence[TaskId] | None = None,
    labels: Sequence[str] | None = ("Before", "After"),
) -> None:
    if task_ids is None:
        task_ids = list(task_stats_dicts[0]["rewards"].keys())

    rewards_list = [[task_stats_dict["rewards"][t] for t in task_ids] for task_stats_dict in task_stats_dicts]

    x = task_ids  # x-axis positions

    fig, ax1 = plt.subplots(figsize=(8, 5))

    for i, rewards in enumerate(rewards_list):
        label = None if labels is None else labels[i]
        ax1.scatter(x, rewards, marker=".", label=label)

    ax1.set_xlabel("Task ID")
    ax1.set_ylabel("Reward")
    ax1.set_title("Rewards per Task")
    ax1.legend()

    plt.tight_layout()
    plt.show()


# def plot_task_comp_stat(
#     task_stats_before: StatsMapping,
#     task_stats_after: StatsMapping,
#     task_ids: Sequence[TaskId] | None = None,
# ) -> None:
#     if task_ids is None:
#         task_ids = list(task_stats_before["rewards"].keys())

#     rewards_before = np.array([task_stats_before["rewards"][t] for t in task_ids])
#     rewards_after = np.array([task_stats_after["rewards"][t] for t in task_ids])
#     counts_before = np.array([task_stats_before["counts"][t] for t in task_ids])
#     counts_after = np.array([task_stats_after["counts"][t] for t in task_ids])
#     steps_before = np.array([task_stats_before["n_steps"][t] for t in task_ids]) / counts_before
#     steps_after = np.array([task_stats_after["n_steps"][t] for t in task_ids]) / counts_after

#     suc_mask_before = rewards_before > 0
#     suc_mask_after = rewards_after > 0

#     # --- Plot 1: Rewards comparison (boxplot) ---
#     plt.figure(figsize=(14, 10))
#     plt.subplot(2, 2, 1)
#     sns.boxplot(data=[rewards_before, rewards_after])
#     plt.xticks([0, 1], ["Before", "After"])
#     plt.ylabel("Rewards")
#     plt.title("Reward Distribution")

#     # --- Plot 2: Steps comparison (boxplot) ---
#     plt.subplot(2, 2, 2)
#     sns.boxplot(data=[steps_before, steps_after])
#     plt.xticks([0, 1], ["Before", "After"])
#     plt.ylabel("Steps")
#     plt.title("Step Distribution")

#     # --- Plot 3: Reward comparison counts ---
#     n_bga = sum(rewards_before > rewards_after)
#     n_bla = sum(rewards_before < rewards_after)
#     n_bea = sum(rewards_before == rewards_after)

#     plt.subplot(2, 2, 3)
#     plt.bar(
#         ["Before > After", "Before < After", "Before = After"],
#         [n_bga, n_bla, n_bea],
#         color=["#4daf4a", "#e41a1c", "#377eb8"],
#     )
#     plt.ylabel("Number of Tasks")
#     plt.title("Reward Comparison")

#     # --- Plot 4: Success overlaps (Venn diagram) ---
#     plt.subplot(2, 2, 4)
#     venn2(
#         subsets=(
#             sum(suc_mask_before & ~suc_mask_after),
#             sum(suc_mask_after & ~suc_mask_before),
#             sum(suc_mask_before & suc_mask_after),
#         ),
#         set_labels=("Before", "After"),
#     )
#     plt.title("Number of solved tasks")

#     plt.tight_layout()
#     plt.show()


def plot_reward_comp(
    task_stats_before: StatsMapping,
    task_stats_after: StatsMapping,
    task_ids: Sequence[TaskId] | None = None,
) -> None:
    if task_ids is None:
        task_ids = list(task_stats_before["rewards"].keys())

    reward_bins = np.arange(17)
    aver_rewards_before = []
    aver_rewards_after = []

    for r in reward_bins:
        task_ids_before = [t for t in task_ids if task_stats_before["rewards"][t] == r]
        if task_ids_before:
            aver_rewards_before.append(r)
            aver_rewards_after.append(np.mean([task_stats_after["rewards"][t] for t in task_ids_before]))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.scatter(aver_rewards_before, aver_rewards_after)
    ax1.plot(reward_bins, reward_bins, "k--")

    ax1.set_xlabel("Average Reward Before")
    ax1.set_ylabel("Average Reward After")
    ax1.grid()

    plt.tight_layout()
    plt.show()
