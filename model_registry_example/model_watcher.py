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
# def build_launch_spec(artifact_version, script_name, api):
#     return {
#         "uri":  "https://github.com/KyleGoyette/wandb_demo_scripts.git",
#         "entry_point": script_name,
#         "overrides": {
#             "artifacts": {
#                 "model": {
#                     "project": project,
#                     "entity": api.entity,
#                     "name": artifact_version.name,
#                     "id": artifact_version.id,
#                     "_version": "v0"
#                 }

#             }
#         }
#     }

class ModelWatcher:
    def __init__(self):
        api = wandb.apis.PublicApi()
        self.api = api
        self.dataset = util.get_eval_dataset_from_wb(api, f"{project}/{model_use_case_id}")
        model_artifact_versions = api.artifact_versions(type_name="model", name=f"{project}/{model_use_case_id}_model_candidates")
        self._arti_versions_len = len(model_artifact_versions)
        self.added_candidates = []

    def enqueue_evals(self, candidates):
        for candidate in candidates:
            if candidate.name not in self.added_candidates:
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
                                }
                            }
                        }
                    },
                    entity=self.api.default_entity,
                    queue="default",
                    resource="local",
                    entry_point="launch_model_evaluator.py",
                    version="main"
                )
            self.added_candidates.append(candidate.name)

    def loop(self):
        while True:
            try:
                candidates = util.get_new_model_candidates_from_wb(project, model_use_case_id, metric_key=f"{self.dataset.name}-ce_loss")
                if len(candidates) > 0:
                    self.enqueue_evals(candidates)
                # artifact_versions = self.api.artifact_versions(type_name="model", name=f"{project}/{model_use_case_id}_model_candidates")
                # if len(artifact_versions) != self._arti_versions_len:
                #     self._arti_versions_len = len(artifact_versions)
                #     print(f"Evaluating new model artifact: {artifact_versions[0].name}")

                time.sleep(3)
            except KeyboardInterrupt:
                print("Closing model watcher...")
                break

x = ModelWatcher()
x.loop()