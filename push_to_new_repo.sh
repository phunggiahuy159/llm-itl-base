#!/usr/bin/env bash

set -e  # exit immediately if any command fails

NEW_REPO_URL="https://github.com/phunggiahuy159/llm-itl-base.git"

echo "Checking git repository..."
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "❌ This directory is not a git repository"
    exit 1
fi

echo "Current remotes:"
git remote -v

echo "Removing old origin (if exists)..."
git remote remove origin 2>/dev/null || true

echo "Adding new origin: $NEW_REPO_URL"
git remote add origin "$NEW_REPO_URL"

echo "Fetching current branch..."
BRANCH=$(git branch --show-current)

if [ -z "$BRANCH" ]; then
    echo "❌ Could not determine current branch"
    exit 1
fi

echo "Current branch: $BRANCH"

echo "Pushing to new repository..."
git push -u origin "$BRANCH"

echo "✅ Done! Repo pushed to $NEW_REPO_URL"
