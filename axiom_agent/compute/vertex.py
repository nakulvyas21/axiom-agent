"""
Vertex AI compute backend - Google Cloud GPU compute (Vertex AI).

The production compute path. Submits a heavy simulation to Google Cloud Vertex
AI as a Custom Job on a GPU-accelerated worker pool, then polls to completion.

Configuration (environment):
  GOOGLE_CLOUD_PROJECT      GCP project id
  GOOGLE_CLOUD_LOCATION     region (default us-central1)
  AXIOM_SIM_IMAGE_URI       container image that runs the simulation engine
  AXIOM_SIM_MACHINE_TYPE    worker machine type (default a2-highgpu-1g)
  AXIOM_SIM_ACCELERATOR     accelerator type  (default NVIDIA_TESLA_A100)

The container itself (the per-domain simulation engine) is a swappable artifact
referenced by image URI - the agent only orchestrates the job.
"""

from __future__ import annotations

import os
import time

from .backend import ComputeBackend, SimulationResult


class VertexComputeBackend(ComputeBackend):
    """Runs simulations as Vertex AI Custom Jobs on GPU workers."""

    name = "vertex-custom-job"

    def __init__(self) -> None:
        self.project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        self.image_uri = os.getenv(
            "AXIOM_SIM_IMAGE_URI",
            "us-docker.pkg.dev/PROJECT/axiom/simulation-engine:latest",
        )
        self.machine_type = os.getenv("AXIOM_SIM_MACHINE_TYPE", "a2-highgpu-1g")
        self.accelerator = os.getenv("AXIOM_SIM_ACCELERATOR", "NVIDIA_TESLA_A100")

    def run_simulation(
        self,
        domain: str,
        spec: str,
        steps: int,
        temperature_k: float,
    ) -> SimulationResult:
        if not self.project:
            return SimulationResult(
                converged=False,
                backend=self.name,
                detail=(
                    "GOOGLE_CLOUD_PROJECT is not set - cannot submit a Vertex AI "
                    "Custom Job. Set it to run on Vertex AI GPU workers."
                ),
            )

        from google.cloud import aiplatform  # imported lazily

        aiplatform.init(project=self.project, location=self.location)

        # A GPU worker pool running the simulation-engine container. The job
        # parameters are passed as container args; the engine writes its
        # converged metrics to the job output.
        worker_pool_specs = [
            {
                "machine_type": self.machine_type,
                "accelerator_type": self.accelerator,
                "accelerator_count": 1,
                "replica_count": 1,
                "container_spec": {
                    "image_uri": self.image_uri,
                    "command": ["python", "run_simulation.py"],
                    "args": [
                        f"--domain={domain}",
                        f"--spec={spec}",
                        f"--steps={steps}",
                        f"--temperature_k={temperature_k}",
                    ],
                },
            }
        ]

        job = aiplatform.CustomJob(
            display_name=f"axiom-{domain}-{steps}-steps",
            worker_pool_specs=worker_pool_specs,
        )
        job.submit()

        # Poll the job to completion. `job.state` reflects the lifecycle;
        # JOB_STATE_SUCCEEDED is terminal-success.
        deadline = time.monotonic() + float(os.getenv("AXIOM_SIM_TIMEOUT_S", "3600"))
        terminal = {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
        }
        while time.monotonic() < deadline:
            state = str(job.state)
            if any(state.endswith(t) for t in terminal):
                break
            time.sleep(10)

        succeeded = str(job.state).endswith("JOB_STATE_SUCCEEDED")
        return SimulationResult(
            converged=succeeded,
            backend=self.name,
            # The simulation-engine container reports the physical metrics; the
            # agent surfaces them once the job lands. Job metadata recorded here.
            metrics={},
            detail=(
                f"Vertex AI Custom Job {job.resource_name} finished with "
                f"state={job.state} on {self.machine_type}/{self.accelerator}."
            ),
        )
