# model_watcher.py
#
# This script represents a trigger system that fires when a new model
# is created
import wandb
from wandb.sdk.launch.launch_add import launch_add
import util
import time


project             = "model_registry_example"
model_use_case_id   = "mnist"
job_type            = "model_trainer"
uri                 = "https://github.com/KyleGoyette/wandb_demo_scripts.git"

class ModelWatcher:
    def __init__(self):
        api = wandb.apis.PublicApi()
        self.api = api
        dataset = util.get_eval_dataset_from_wb(api, f"{project}/{model_use_case_id}")
        self.dataset_version = dataset.version
        model_artifact_versions = api.artifact_versions(type_name="model",
                                                        name=f"{project}/{model_use_case_id}_model_candidates")
        self._arti_versions_len = len(model_artifact_versions)
        self.all_model_candidates = {}
        for a in model_artifact_versions:
            self.all_model_candidates[a.name] = a

        unevaluated_model_candidates = util.get_new_model_candidates_from_wb(
                    project, 
                    model_use_case_id,
                    metric_key=f"{dataset.name}-ce_loss"
                )
        self.enqueue_model_evals(unevaluated_model_candidates, dataset)

    def enqueue_model_evals(self, candidates, dataset):
        for candidate in candidates:
            launch_add(
                uri,
                project=project,
                config={
                    "overrides": { 
                        "artifacts": {
                            "model": {
                                "project": project,
                                "entity": self.api.default_entity,
                                "name": candidate.name,
                                "id": candidate.id,
                                "_version": "v0"
                            },
                            "dataset": {
                                "project": project,
                                "entity": self.api.default_entity,
                                "name": dataset.name,
                                "id": dataset.id,
                                "_version": "v0"
                            }
                        }
                    }
                },
                entity=self.api.default_entity,
                queue="default",
                resource="local",
                entry_point="model_registry_example/launch_model_evaluator.py",
                version="main"
            )
            self.all_model_candidates[candidate.name] = candidate

    def loop(self):
        while True:
            try:
                dataset = util.get_eval_dataset_from_wb(self.api, f"{project}/{model_use_case_id}")

                print("Checking for new datasets to evaluate models on...")
                # check for new dataset, if so test all candidates
                if dataset.version != self.dataset_version:
                    candidates = [item for _, item in self.all_model_candidates.keys()]
                    self.enqueue_model_evals(candidates, dataset)
                    self.dataset_version = dataset.version
                    continue

                print("Checking for new model candidates to evaluate on latest dataset")
                # check for unevaluated model candidates, test against latest dataset
                candidates = util.get_new_model_candidates_from_wb(
                    project, 
                    model_use_case_id,
                    metric_key=f"{dataset.name}-ce_loss"
                )
                if len(candidates) > 0:
                    unqueued_candidates = [c for c in candidates if c.name not in self.all_model_candidates.keys()]
                    self.enqueue_model_evals(unqueued_candidates, dataset)

                time.sleep(3)
            except KeyboardInterrupt:
                print("Closing model watcher...")
                break

x = ModelWatcher()
x.loop()