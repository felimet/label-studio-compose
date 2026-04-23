#!/bin/sh
# MinIO initialisation script
# Creates buckets, sets anonymous download policy, provisions a
# least-privilege service account for Label Studio, and optionally
# sets a bucket quota.
# Idempotent — safe to re-run.
#
# MINIO_BUCKET supports comma-separated bucket names:
#   MINIO_BUCKET=default-bucket,test,another-bucket
set -eu

MINIO_ALIAS="local"
MINIO_ENDPOINT="http://minio:9000"
BUCKETS="${MINIO_BUCKET:-label-studio-bucket}"
ANONYMOUS_DOWNLOAD="${MINIO_ANONYMOUS_DOWNLOAD:-false}"

echo "[init-minio] Waiting for MinIO to be ready..."
until mc alias set "${MINIO_ALIAS}" "${MINIO_ENDPOINT}" \
        "${MINIO_ROOT_USER}" "${MINIO_ROOT_PASSWORD}" 2>/dev/null; do
    sleep 2
done
echo "[init-minio] Connected to MinIO."

# ── Create buckets ──────────────────────────────────────────
for BUCKET in $(echo "${BUCKETS}" | tr ',' ' '); do
    echo "[init-minio] Creating bucket: ${BUCKET}"
    mc mb --ignore-existing "${MINIO_ALIAS}/${BUCKET}"

    # Policy defaults to private for better security.
    # Set MINIO_ANONYMOUS_DOWNLOAD=true only when public read is required.
    case "${ANONYMOUS_DOWNLOAD}" in
        true|TRUE|True|1|yes|YES|Yes)
            echo "[init-minio] Setting anonymous download policy on ${BUCKET}"
            mc anonymous set download "${MINIO_ALIAS}/${BUCKET}"
            ;;
        *)
            echo "[init-minio] Setting private policy on ${BUCKET}"
            mc anonymous set none "${MINIO_ALIAS}/${BUCKET}"
            ;;
    esac
done

# ── CORS ──────────────────────────────────────────────────
# MinIO open-source editions >= 2024 removed the S3 PutBucketCors API.
# CORS is now controlled via MINIO_API_CORS_ALLOW_ORIGIN env var (set in
# docker-compose.yml on the minio service) — no mc command needed here.
echo "[init-minio] CORS handled via MINIO_API_CORS_ALLOW_ORIGIN server env var."

# ── Service account for Label Studio ──────────────────────
# Provision a dedicated access key scoped to all configured buckets.
# Label Studio should use MINIO_LS_ACCESS_ID / MINIO_LS_SECRET_KEY
# (not the root credentials) when configuring Cloud Storage in the UI.
if [ -n "${MINIO_LS_ACCESS_ID:-}" ] && [ -n "${MINIO_LS_SECRET_KEY:-}" ]; then
    echo "[init-minio] Provisioning Label Studio service account..."

    # Build JSON resource arrays for IAM policy
    OBJECT_RESOURCES=""
    BUCKET_RESOURCES=""
    FIRST=1
    for BUCKET in $(echo "${BUCKETS}" | tr ',' ' '); do
        if [ "${FIRST}" -eq 1 ]; then
            OBJECT_RESOURCES="\"arn:aws:s3:::${BUCKET}/*\""
            BUCKET_RESOURCES="\"arn:aws:s3:::${BUCKET}\""
            FIRST=0
        else
            OBJECT_RESOURCES="${OBJECT_RESOURCES}, \"arn:aws:s3:::${BUCKET}/*\""
            BUCKET_RESOURCES="${BUCKET_RESOURCES}, \"arn:aws:s3:::${BUCKET}\""
        fi
    done

        # Write bucket-scoped IAM policy.
        # tus resumable upload relies on S3 multipart APIs; grant the
        # multipart-specific actions explicitly to avoid AccessDenied.
    cat > /tmp/ls-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:AbortMultipartUpload",
                "s3:ListMultipartUploadParts"
            ],
      "Resource": [${OBJECT_RESOURCES}]
    },
    {
      "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetBucketLocation",
                "s3:ListBucketMultipartUploads"
            ],
      "Resource": [${BUCKET_RESOURCES}]
    }
  ]
}
EOF

    # Remove existing policy then recreate (idempotent update)
    mc admin policy remove "${MINIO_ALIAS}" ls-bucket-policy 2>/dev/null || true
    mc admin policy create "${MINIO_ALIAS}" ls-bucket-policy /tmp/ls-policy.json

    # Create user (skip if already exists)
    mc admin user info "${MINIO_ALIAS}" "${MINIO_LS_ACCESS_ID}" > /dev/null 2>&1 || \
        mc admin user add "${MINIO_ALIAS}" "${MINIO_LS_ACCESS_ID}" "${MINIO_LS_SECRET_KEY}"

    # Attach policy to user (idempotent)
    mc admin policy attach "${MINIO_ALIAS}" ls-bucket-policy \
        --user "${MINIO_LS_ACCESS_ID}" 2>/dev/null || true

    echo "[init-minio] Service account ready: ${MINIO_LS_ACCESS_ID}"
    echo "[init-minio] Buckets covered: ${BUCKETS}"
    echo "[init-minio] Use this key when configuring Label Studio Cloud Storage (S3)."
else
    echo "[init-minio] MINIO_LS_ACCESS_ID/SECRET_KEY not set — skipping service account."
    echo "[init-minio] WARNING: Label Studio will use root credentials for storage access."
fi

# ── Bucket quota (optional) ───────────────────────────────
if [ -n "${MINIO_BUCKET_QUOTA_GB:-}" ]; then
    for BUCKET in $(echo "${BUCKETS}" | tr ',' ' '); do
        echo "[init-minio] Setting bucket quota on ${BUCKET}: ${MINIO_BUCKET_QUOTA_GB} GiB"
        mc quota set "${MINIO_ALIAS}/${BUCKET}" --size "${MINIO_BUCKET_QUOTA_GB}GiB"
    done
fi

# ── Final verification ─────────────────────────────────────
echo "[init-minio] Verifying buckets..."
for BUCKET in $(echo "${BUCKETS}" | tr ',' ' '); do
    mc ls "${MINIO_ALIAS}/${BUCKET}" && echo "[init-minio] Bucket OK: ${BUCKET}"
done
echo "[init-minio] Done."
