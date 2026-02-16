#!/bin/bash
# Grant GCS permissions so Veo extend works with bucket gs://veo3downloadcvgenix
#
# OPTION A - Google Cloud Console (no gcloud needed):
#   1. Open https://console.cloud.google.com/storage/browser/veo3downloadcvgenix
#   2. Click the bucket "veo3downloadcvgenix" -> Permissions tab -> Grant access
#   3. Add BOTH of these principals (one at a time or together with same role):
#      - automation-site@automationproject-486823.iam.gserviceaccount.com  Role: Storage Object Viewer
#      - service-779092894543@gcp-sa-aiplatform.iam.gserviceaccount.com    Role: Storage Object Creator
#   4. Save
#
# OPTION B - If you have gcloud/gsutil installed:
#   Run: ./setup-gcs-permissions.sh

set -e
BUCKET="gs://veo3downloadcvgenix"
PROJECT_ID="automationproject-486823"

if ! command -v gsutil &>/dev/null; then
  echo "gsutil not found. Use Google Cloud Console (Option A in this script's comments)."
  echo ""
  echo "Add these two principals to bucket veo3downloadcvgenix Permissions:"
  echo "  1. automation-site@${PROJECT_ID}.iam.gserviceaccount.com  -> Storage Object Viewer"
  echo "  2. service-779092894543@gcp-sa-aiplatform.iam.gserviceaccount.com -> Storage Object Creator"
  exit 1
fi

APP_SA="automation-site@${PROJECT_ID}.iam.gserviceaccount.com"
echo "Granting Storage Object Viewer (read) to ${APP_SA} on ${BUCKET}..."
gsutil iam ch "serviceAccount:${APP_SA}:objectViewer" "${BUCKET}"

VERTEX_SA="service-779092894543@gcp-sa-aiplatform.iam.gserviceaccount.com"
echo "Granting Storage Object Creator (write) to Vertex AI service agent on ${BUCKET}..."
gsutil iam ch "serviceAccount:${VERTEX_SA}:objectCreator" "${BUCKET}"

echo "Done. Both accounts now have access to ${BUCKET}."
