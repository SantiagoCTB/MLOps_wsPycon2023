import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()
if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return TensorDataset(x, y)

def preprocess_and_log(steps):
    with wandb.init(project="MLOps-Pycon2023",
                    name=f"Preprocess Diabetes ExecId-{args.IdExecution}",
                    job_type="preprocess-data") as run:

        raw_art = run.use_artifact('diabetes-raw:latest')
        data_dir = raw_art.download()

        train_ds = read(data_dir, "train")
        val_ds   = read(data_dir, "valid")
        test_ds  = read(data_dir, "test")

        # Fit scalers en TRAIN
        X_train, y_train = train_ds.tensors
        X_val,   y_val   = val_ds.tensors
        X_test,  y_test  = test_ds.tensors

        x_scaler = StandardScaler()
        X_train_s = torch.tensor(x_scaler.fit_transform(X_train), dtype=torch.float32)
        X_val_s   = torch.tensor(x_scaler.transform(X_val),       dtype=torch.float32)
        X_test_s  = torch.tensor(x_scaler.transform(X_test),      dtype=torch.float32)

        y_scaler = None
        if steps.get("scale_y", False):
            y_scaler = StandardScaler()
            y_train_s = torch.tensor(y_scaler.fit_transform(y_train), dtype=torch.float32)
            y_val_s   = torch.tensor(y_scaler.transform(y_val),       dtype=torch.float32)
            y_test_s  = torch.tensor(y_scaler.transform(y_test),      dtype=torch.float32)
        else:
            y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

        processed = {
            "train": TensorDataset(X_train_s, y_train_s),
            "valid": TensorDataset(X_val_s,   y_val_s),
            "test":  TensorDataset(X_test_s,  y_test_s)
        }

        processed_art = wandb.Artifact(
            "diabetes-preprocess", type="dataset",
            description="Preprocessed Diabetes dataset (scaled)",
            metadata={"steps": steps, "x_scaler": "StandardScaler", "y_scaled": bool(steps.get("scale_y", False))}
        )

        for name, ds in processed.items():
            with processed_art.new_file(name + ".pt", mode="wb") as f:
                x, y = ds.tensors
                torch.save((x, y), f)

        run.log_artifact(processed_art)

if __name__ == "__main__":
    steps = {"scale_y": False}
    preprocess_and_log(steps)
