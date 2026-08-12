#!/bin/bash

set -e

echo "🔍 Checking DVC before Git push..."

if ! command -v dvc >/dev/null 2>&1; then
    echo "❌ DVC is not available."
    echo "   Activate your DVC environment first."
    exit 1
fi

echo ""
echo "📋 Checking DVC pipeline status..."

DVC_STATUS=$(dvc status 2>&1) || {
    echo "$DVC_STATUS"
    echo ""
    echo "❌ DVC status check failed."
    echo "   Git push blocked."
    exit 1
}

if [ -n "$DVC_STATUS" ]; then
    echo "$DVC_STATUS"
    echo ""
    echo "❌ DVC pipeline has changed outputs."
    echo "   Git push blocked."
    echo ""
    echo "Run:"
    echo "    dvc repro"
    echo ""
    echo "Then commit the resulting changes."
    exit 1
fi

echo "✅ DVC pipeline is up to date."

echo ""
echo "☁️  Checking DVC remote..."

DVC_CLOUD_STATUS=$(dvc status --cloud 2>&1) || {
    echo "$DVC_CLOUD_STATUS"
    echo ""
    echo "❌ DVC remote check failed."
    echo "   Git push blocked."
    echo ""
    echo "Run:"
    echo "    dvc push"
    exit 1
}

echo "$DVC_CLOUD_STATUS"

echo ""
echo "✅ DVC remote is synchronized."
echo "🚀 Git push allowed."
exit 0
