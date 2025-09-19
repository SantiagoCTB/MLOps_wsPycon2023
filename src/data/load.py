import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import argparse
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()
if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=0.8, val_size=0.1, random_state=42):
    data = load_diabetes()
    X = data.data  # (n_samples, 10)
    y = data.target  # (n_samples,)

    # split train / temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    # split temp into val/test
    val_ratio = val_size / (1.0 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - val_ratio, random_state=random_state
    )

    # to tensors
    def to_tensor_ds(Xa, ya):
        Xt = torch.tensor(Xa, dtype=torch.float32)
        yt = torch.tensor(ya, dtype=torch.float32).unsqueeze(1)  # (n,1)
        return TensorDataset(Xt, yt)

    train_ds = to_tensor_ds(X_train, y_train)
    val_ds   = to_tensor_ds(X_val,   y_val)
    test_ds  = to_tensor_ds(X_test,  y_test)

    return (train_ds, val_ds, test_ds)

def load_and_log():
    with wandb.init(project="MLOps-Pycon2023",
                    name=f"Load Diabetes ExecId-{args.IdExecution}",
                    job_type="load-data") as run:
        raw_data = wandb.Artifact(
            "diabetes-raw", type="dataset",
            description="Raw Diabetes dataset (tabular)",
            metadata={"source": "sklearn.datasets.load_diabetes"}
        )

        train_ds, val_ds, test_ds = load()
        names = ["train", "valid", "test"]
        datasets = [train_ds, val_ds, test_ds]

        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        run.log_artifact(raw_data)

# Ejecutar si se llama como script
if __name__ == "__main__":
    load_and_log()
