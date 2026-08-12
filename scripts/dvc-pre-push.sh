#!/bin/bash

set -e

echo "🔍 Checking DVC before Git push..."
echo ""

if ! command -v dvc >/dev/null 2>&1; then
    echo "❌ DVC is not available."
    echo "   Activate your DVC environment first."
    exit 1
fi

echo "☁️ Checking DVC remote..."

DVC_CLOUD_STATUS=$(dvc status --cloud 2>&1)
DVC_EXIT=$?

echo "$DVC_CLOUD_STATUS"

if [ "$DVC_EXIT" -ne 0 ]; then
    echo ""
    echo "❌ Unable to verify DVC remote."
    echo "   Git push blocked."
    echo ""
    echo "Run:"
    echo "    dvc push"
    echo ""
    echo "Then retry:"
    echo "    git push"
    exit 1
fi

# DVC status --cloud should report that the local cache and
# configured remote are synchronized when everything has been pushed.
if echo "$DVC_CLOUD_STATUS" | grep -q "in sync"; then
    echo ""
    echo "✅ DVC cache and remote are synchronized."
    echo "🚀 Git push allowed."
    exit 0
fi

echo ""
echo "❌ DVC remote is not synchronized."
echo "   Git push blocked."
echo ""
echo "Run:"
echo "    dvc push"
echo ""
echo "Then retry:"
echo "    git push"

exit 1