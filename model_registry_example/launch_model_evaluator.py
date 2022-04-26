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
metric=f"{dataset.name}-ce_loss"
model_name = "{}_model_candidates:v0".format(model_use_case_id)
model_artifact, score = util.evaluate_model_by_name(model_name, x_eval, y_eval)
util.save_metric_to_model_in_wb(model_artifact, metric, score)

# Finally, promote the best model to production.
#util.promote_best_model_in_wb(project, model_use_case_id, metric)
