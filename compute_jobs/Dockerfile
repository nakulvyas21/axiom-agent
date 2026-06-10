# Simulation engine container for Vertex AI Custom Jobs (GPU workers).
#
# The reference entrypoint is pure-Python, so a slim base builds and runs
# anywhere. For a production engine, switch the base image to a CUDA runtime
# (e.g. nvidia/cuda:12.4.0-runtime) and install the per-domain engines here.
FROM python:3.11-slim

WORKDIR /app
COPY run_simulation.py /app/run_simulation.py

# Per-domain engine dependencies would be installed here, e.g.:
#   RUN pip install --no-cache-dir openmm
# The reference integrator needs only the standard library.

ENTRYPOINT ["python", "run_simulation.py"]
