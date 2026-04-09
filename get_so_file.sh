#!/usr/bin/env bash
# usage: ./get_so_file.sh [dest-dir]
set -euo pipefail

DEST="${1:-./mp-out}"

echo "Extracting artifacts from local build to $DEST..."

mkdir -p "$DEST/plugins"

if [ -f "build/libgstliveportrait.so" ]; then
    cp "build/libgstliveportrait.so" "$DEST/plugins/liveportrait_gst.so"
    echo "Copied build/libgstliveportrait.so to $DEST/plugins/liveportrait_gst.so"
else
    echo "Error: build/libgstliveportrait.so not found. Please build the plugin first."
    exit 1
fi

echo "Wrote:"
ls -l "$DEST/plugins"
