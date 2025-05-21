# This script collects the most vital results of all experiments found in EXPERIMENT_PATH to a CSV located at OUT_CSV_PATH
import os.path as pt
import os
import json
import pandas as pd
import numpy as np

EXPERIMENT_PATH=pt.expanduser("~/TimeSeAD/log/grid_search")
OUT_CSV_PATH=pt.expanduser("~/TimeSeAD/overview.csv")


if __name__ == '__main__':
    df = pd.DataFrame(columns=["Experiment", "status", "start_time", "stop_time", "auc", "best_f1_score", "auprc", "Folder"])

    for folder in sorted(
            os.listdir(EXPERIMENT_PATH),
            key=lambda s: (f"{int(s.strip()):06d}" if s.strip().isdigit() else s)
        ):
        row = {col: "" for col in df.columns}
        if folder == "_sources":
            continue
        config_file = pt.join(EXPERIMENT_PATH, folder, "config.json")
        info_file = pt.join(EXPERIMENT_PATH, folder, "info.json")
        run_file = pt.join(EXPERIMENT_PATH, folder, "run.json")
        if not (pt.exists(config_file) and pt.getsize(config_file) > 0):
            continue
        with open(config_file, 'r') as reader:
            config = json.load(reader)
            row["Experiment"] = config["params"]["training_experiment"]
        if pt.exists(info_file) and pt.getsize(info_file) > 0:
            with open(info_file, 'r') as reader:
                info = json.load(reader)
            if "final_scores" in info:
                row["status"] = "done"
                row["auc"] = str(info["final_scores"]["auc"]["value"])
                row["best_f1_score"] = str(info["final_scores"]["best_f1_score"])
                row["auprc"] = str(info["final_scores"]["auprc"])
        if pt.exists(run_file) and pt.getsize(run_file) > 0:
            with open(run_file, 'r') as reader:
                run = json.load(reader)
            row["status"] = run.get("status", row["status"])
            row["start_time"] = run.get("start_time", row["start_time"])
            row["stop_time"] = run.get("stop_time", row["stop_time"])
        row["Folder"] = folder
        df = df.append(row, ignore_index=True)

    df.to_csv(OUT_CSV_PATH, index=False)