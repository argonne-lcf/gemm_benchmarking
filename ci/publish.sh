#!/bin/bash
set -euo pipefail

RESULTS=${1:?usage: publish.sh <results-dir>}
GITHUB_URL=git@github.com:argonne-lcf/gemm_benchmarking.git
DEST=aurora_$(date +%Y/%m)
BRANCH=results/$(date +%Y-%m)

git fetch "$GITHUB_URL" main
git checkout -B "$BRANCH" FETCH_HEAD

mkdir -p "$DEST"
cp "$RESULTS"/data.txt \
   "$RESULTS"/output_no_post_process "$RESULTS"/output_post_process \
   "$RESULTS"/output_no_post_process.png "$RESULTS"/output_post_process.png \
   "$DEST"/

git add "$DEST"
git -c user.name="gemm CI" -c user.email="noreply@alcf.anl.gov" \
    commit -m "adding $(date +%B | tr 'A-Z' 'a-z')"
git push --force-with-lease "$GITHUB_URL" "$BRANCH"
