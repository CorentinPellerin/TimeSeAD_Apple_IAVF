# This script just evaluates finished subruns (e.g., subrun '2' for run 'train_iforest')
import os.path as pt
import os
import json
import pandas as pd
import numpy as np

LOG_PATH=pt.expanduser("~/TimeSeAD/log/")
RUN_FILE=pt.expanduser("~/TimeSeAD/timesead_experiments/grid_search.py")
TRAINING_FILE_FOLDER=pt.expanduser("~/TimeSeAD/experiment_configs/apple/")

if __name__ == '__main__':
    possible_training_files = []
    for folder in os.listdir(TRAINING_FILE_FOLDER):
        if folder == "experiment_manager":
            continue
        for file in os.listdir(pt.join(TRAINING_FILE_FOLDER, folder)):
            possible_training_files.append(pt.join(TRAINING_FILE_FOLDER, folder, file))

    todo = []
    for superfolder in os.listdir(LOG_PATH):
        if superfolder == "grid_search":
            continue
        experiment_path = pt.join(LOG_PATH, superfolder)
        found_training_files = [f for f in possible_training_files if superfolder in f]
        if len(found_training_files) != 1:
            raise ValueError(f"Did not find matching training file for {superfolder}!")
        training_file = found_training_files[0]
        train_ids = []
        for folder in sorted(
                os.listdir(experiment_path),
                key=lambda s: (f"{int(s.strip()):06d}" if s.strip().isdigit() else s)
            ):
            if folder == "_sources":
                continue
            config_file = pt.join(experiment_path, folder, "config.json")
            info_file = pt.join(experiment_path, folder, "info.json")
            run_file = pt.join(experiment_path, folder, "run.json")
            if not pt.exists(config_file):
                continue
            with open(config_file, 'r') as reader:
                config = json.load(reader)
            if pt.exists(info_file):
                continue
            if pt.exists(run_file):
                with open(run_file, 'r') as reader:
                    run = json.load(reader)
            if pt.exists(pt.join(LOG_PATH, experiment_path, folder, "final_model.pth")):
                train_ids.append(folder)

        if len(train_ids) > 0:
            todo.append(f"python {RUN_FILE} with {training_file} training_param_updates.training.device=cuda seed=123 params.train_ids={','.join(train_ids)}")

print(f"bash run_screen_timesead.sh -m 0.95 \\")
for i, t in enumerate(todo):
    end = "\\" if i < (len(todo) - 1) else ""
    print(f'"{t}" {end}')
print()