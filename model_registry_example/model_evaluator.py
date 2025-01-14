# model_evaluator.py
#
# This script represents a workload that evaluates all models which
# have not yet been evaluated on the latest evaluation set. Moreover,
# it will also tag the best model with "production" so that other systems
# can use it for inference.
import os
import wandb
import util
import argparse

project             = "launch_demo_april_2022"
model_use_case_id   = "mnist"
job_type            = "evaluator"

# First, we launch a run which registers this workload with WandB.
run = wandb.init(project=project, job_type=job_type)

# Then we fetch the latest evaluation set.
x_eval, y_eval, dataset = util.download_eval_dataset_from_wb(model_use_case_id)

# Next we fetch the new candidate models for this use case
metric=f"{dataset.name}-ce_loss"
print(metric)
candidates = util.get_new_model_candidates_from_wb(project, model_use_case_id, metric)

# Evaluate the models and save their metrics to wb.
for model in candidates:
    _, score = util.evaluate_model(model, x_eval, y_eval)
    util.save_metric_to_model_in_wb(model, metric, score)


