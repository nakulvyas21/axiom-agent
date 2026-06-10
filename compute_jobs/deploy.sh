#!/usr/bin/env bash
# Build and push the simulation-engine container to Artifact Registry, so the
# Vertex backend can launch it as a GPU Custom Job. Idempotent; safe to re-run.
#
# Prereqs: gcloud authenticated, a GCP project with billing, Artifact Registry
# and Vertex AI APIs enabled.
#
# Usage:
#   GOOGLE_CLOUD_PROJECT=my-project ./deploy.sh
#
# Then point the agent at it:
#   export AXIOM_COMPUTE_BACKEND=vertex
#   export AXIOM_SIM_IMAGE_URI=<the IMAGE_URI printed below>

set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
LOCATION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
REPO="${AXIOM_AR_REPO:-axiom}"
IMAGE_URI="${LOCATION}-docker.pkg.dev/${PROJECT}/${REPO}/simulation-engine:latest"

echo "Ensuring Artifact Registry repo '${REPO}' exists..."
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${LOCATION}" --project="${PROJECT}" \
  2>/dev/null || true

echo "Building and pushing ${IMAGE_URI}..."
gcloud builds submit --tag "${IMAGE_URI}" --project="${PROJECT}" "$(dirname "$0")"

echo
echo "Done. To enable Google Cloud GPU compute:"
echo "  export AXIOM_COMPUTE_BACKEND=vertex"
echo "  export AXIOM_SIM_IMAGE_URI=${IMAGE_URI}"
