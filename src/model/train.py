import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import os
import argparse
import wandb
from src.Classifier import Regressor

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return TensorDataset(x, y)

def r2_score_torch(y_true, y_pred):
    # y_true, y_pred: (n,1)
    y_true_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_true_mean) ** 2)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)

def train_and_log(config, experiment_id="0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with wandb.init(project="MLOps-Pycon2023",
                    name=f"Train Diabetes ExecId-{experiment_id}",
                    job_type="train-model",
                    config=config) as run:

        data_art = run.use_artifact('diabetes-preprocess:latest')
        data_dir = data_art.download()

        train_ds = read(data_dir, "train")
        valid_ds = read(data_dir, "valid")

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=config.batch_size)

        model = Regressor(config.input_shape, config.hidden_layer_1, config.hidden_layer_2).to(device)
        optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.lr)
        criterion = nn.MSELoss()

        example_ct = 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)  # (N,10), (N,1)

                optimizer.zero_grad()
                output = model(data)             # (N,1)
                loss = criterion(output, target) # MSE
                loss.backward()
                optimizer.step()

                example_ct += len(data)

                if batch_idx % config.batch_log_interval == 0:
                    wandb.log({"train/mse": loss.item(),
                               "epoch": epoch,
                               "seen_examples": example_ct})

            # Validación
            model.eval()
            val_loss, val_rmse, val_r2 = 0.0, 0.0, 0.0
            n = 0
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    pred = model(data)
                    mse = criterion(pred, target)
                    rmse = torch.sqrt(mse)
                    r2 = r2_score_torch(target, pred)

                    bs = data.shape[0]
                    n += bs
                    val_loss += mse.item() * bs
                    val_rmse += rmse.item() * bs
                    val_r2 += r2.item() * bs

            val_loss /= n
            val_rmse /= n
            val_r2   /= n

            wandb.log({"valid/mse": val_loss, "valid/rmse": val_rmse, "valid/r2": val_r2, "epoch": epoch})
            print(f"Epoch {epoch}: val_mse={val_loss:.4f} | val_rmse={val_rmse:.4f} | val_r2={val_r2:.4f}")

        # Guardar modelo entrenado como artifact
        model_art = wandb.Artifact(
            "diabetes-model", type="model",
            description="Trained MLP Regressor",
            metadata=dict(config)
        )
        torch.save(model.state_dict(), "trained_model.pth")
        model_art.add_file("trained_model.pth")
        wandb.save("trained_model.pth")
        run.log_artifact(model_art)

        return model

def evaluate_and_log(experiment_id='0', config=None):
    with wandb.init(project="MLOps-Pycon2023",
                    name=f"Eval Diabetes ExecId-{experiment_id}",
                    job_type="eval-model",
                    config=config) as run:

        data_art = run.use_artifact('diabetes-preprocess:latest')
        data_dir = data_art.download()
        test_ds = read(data_dir, "test")
        test_loader = DataLoader(test_ds, batch_size=config.batch_size)

        # Cargar el modelo entrenado si prefieres (o re-instanciar)
        model_art = run.use_artifact('diabetes-model:latest')
        model_dir = model_art.download()
        model = Regressor(config.input_shape, config.hidden_layer_1, config.hidden_layer_2)
        model.load_state_dict(torch.load(os.path.join(model_dir, "trained_model.pth"), map_location="cpu"))
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        criterion = nn.MSELoss()

        test_mse, test_rmse, test_r2, n = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                pred = model(data)
                mse = criterion(pred, target)
                rmse = torch.sqrt(mse)
                r2 = r2_score_torch(target, pred)

                bs = data.shape[0]
                n += bs
                test_mse += mse.item() * bs
                test_rmse += rmse.item() * bs
                test_r2 += r2.item() * bs

        test_mse /= n
        test_rmse /= n
        test_r2   /= n

        wandb.log({"test/mse": test_mse, "test/rmse": test_rmse, "test/r2": test_r2})
        print(f"TEST: mse={test_mse:.4f} | rmse={test_rmse:.4f} | r2={test_r2:.4f}")

if __name__ == "__main__":
    # ejemplo de config
    train_config = {
        "input_shape": 10,
        "hidden_layer_1": 64,
        "hidden_layer_2": 64,
        "batch_size": 64,
        "epochs": 50,
        "batch_log_interval": 10,
        "optimizer": "Adam",
        "lr": 1e-3
    }
    # ENTRENAR
    train_and_log(train_config, experiment_id=args.IdExecution or "local")
    # EVALUAR
    evaluate_and_log(experiment_id=args.IdExecution or "local", config=train_config)
