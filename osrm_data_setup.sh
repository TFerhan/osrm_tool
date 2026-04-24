#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OSRM Setup — Car (port 5000) + Foot (port 5001)
# Run this from the directory containing morocco-latest.osm.pbf
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

wget https://download.geofabrik.de/africa/morocco-latest.osm.pbf

PBF="morocco-latest.osm.pbf"
IMG="ghcr.io/project-osrm/osrm-backend"

# ── 1. Separate working dirs ──────────────────────────────────────────────────
mkdir -p car foot

# Symlink (or copy if on different filesystems) the PBF into each dir
# Using hard link to avoid doubling disk usage
if [ ! -f "car/$PBF" ];  then ln "$PBF" "car/$PBF"  2>/dev/null || cp "$PBF" "car/$PBF";  fi
if [ ! -f "foot/$PBF" ]; then ln "$PBF" "foot/$PBF" 2>/dev/null || cp "$PBF" "foot/$PBF"; fi

# ── 2. Extract ────────────────────────────────────────────────────────────────
echo ">>> Extracting car profile …"
docker run --rm -t -v "${PWD}/car:/data" "$IMG" \
    osrm-extract -p /opt/car.lua /data/"$PBF"

echo ">>> Extracting foot profile …"
docker run --rm -t -v "${PWD}/foot:/data" "$IMG" \
    osrm-extract -p /opt/foot.lua /data/"$PBF"

# ── 3. Contract (CH algorithm — fastest query speed, ~10× faster than MLD) ───
echo ">>> Contracting car …"
docker run --rm -t -v "${PWD}/car:/data" "$IMG" \
    osrm-contract /data/morocco-latest.osrm

echo ">>> Contracting foot …"
docker run --rm -t -v "${PWD}/foot:/data" "$IMG" \
    osrm-contract /data/morocco-latest.osrm


echo ">>> Starting osrm-car  on :5000 …"
docker run -d --rm --name osrm-car \
    -p 5000:5000 \
    -v "${PWD}/car:/data" \
    "$IMG" \
    osrm-routed \
        --algorithm ch \
        --max-table-size 8000 \
        --threads 2 \
        /data/morocco-latest.osrm

echo ">>> Starting osrm-foot on :5001 …"
docker run -d --rm --name osrm-foot \
    -p 5001:5000 \
    -v "${PWD}/foot:/data" \
    "$IMG" \
    osrm-routed \
        --algorithm ch \
        --max-table-size 8000 \
        --threads 2 \
        /data/morocco-latest.osrm

echo ""
echo "✓ Both OSRM servers running."
echo "  car  → http://localhost:5000/table/v1/driving/"
echo "  foot → http://localhost:5001/table/v1/driving/"
echo ""
echo "Quick health check:"
curl -s "http://localhost:5000/health" && echo " ← car OK"
curl -s "http://localhost:5001/health" && echo " ← foot OK"
