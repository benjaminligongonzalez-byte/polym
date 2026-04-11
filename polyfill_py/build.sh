#!/usr/bin/env bash
# Build and install the polyfill-py Rust extension.
#
# Usage:
#   ./polyfill_py/build.sh          # release build (recommended)
#   ./polyfill_py/build.sh --debug  # debug build (faster compile, slower runtime)
#
# After this runs, "from polyfill_py import PolyRustClient" will work.
# Then launch the bot with:  python bot.py --rust --paper

set -euo pipefail
cd "$(dirname "$0")"

# Install build dependencies if missing
pip install maturin patchelf --quiet

if [[ "${1:-}" == "--debug" ]]; then
    echo "[build.sh] Building debug wheel…"
    maturin build
else
    echo "[build.sh] Building optimised release wheel…"
    maturin build --release
fi

WHEEL=$(ls -t target/wheels/polyfill_py-*.whl | head -1)
echo "[build.sh] Installing: $WHEEL"
pip install "$WHEEL" --force-reinstall --quiet

python3 -c "from polyfill_py import PolyRustClient; print('[build.sh] ✓ polyfill_py installed successfully')"
