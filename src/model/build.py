import torch
import os
import argparse
import wandb


# Ojo con el import: si renombraste a Regressor:
from src.model.src.Classifier import Regressor  # ajusta el path real

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()
if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def build_model_and_log(config, model, model_name="MLPRegressor", model_description="Simple MLP Regressor"):
    with wandb.init(project="MLOps-Pycon2023",
                    name=f"initialize Model ExecId-{args.IdExecution}",
                    job_type="initialize-model",
                    config=config) as run:

        model_artifact = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config)
        )

        os.makedirs("./model", exist_ok=True)
        name_artifact_model = f"initialized_model_{model_name}.pth"
        torch.save(model.state_dict(), f"./model/{name_artifact_model}")
        model_artifact.add_file(f"./model/{name_artifact_model}")
        wandb.save(name_artifact_model)

        run.log_artifact(model_artifact)

if __name__ == "__main__":
    # Diabetes: 10 features
    input_shape = 10
    model_config = {"input_shape": input_shape,
                    "hidden_layer_1": 64,
                    "hidden_layer_2": 64}
    model = Regressor(**model_config)
    build_model_and_log(model_config, model, "MLPRegressor", "Simple MLP Regressor")
