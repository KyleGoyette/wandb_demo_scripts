# model_trainer.py
#
# This script represents a workload that trains a model based on the
# latest training data set for a given modelling use case.

import wandb
import util
import argparse
import os

project             = "launch_demo_april_2022"
model_use_case_id   = "mnist"
job_type            = "model_trainer"

# First, we launch a run which registers this workload with WandB.
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',         type=int,   default=128)
parser.add_argument('--epochs',             type=int,   default=5)
parser.add_argument('--optimizer',          type=str,   default="adam")
parser.add_argument('--validation_split',   type=float, default=0.1)
args = parser.parse_args()
run = wandb.init(project=project, job_type=job_type, config=args)

# Next we download the latest training data available for this use case from WandB. 
# Again, the domain specific logic is abstracted away in a helper function.
x_train, y_train = util.download_training_dataset_from_wb(model_use_case_id)

# Then we train a model using this data. For simplicity, we use a sequential model.
model = util.build_and_train_model(x_train, y_train, config=run.config)
model.save(f"/opt/ml/model/{model_use_case_id}_model.h5")

# Finally, we publish the model to WandB. This will create a new artifact version
# that serves as a "candidate" model for this use case.
util.publish_model_candidate_to_wb(model, model_use_case_id)

