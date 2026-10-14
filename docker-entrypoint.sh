#!/usr/bin/env bash
set -euo pipefail

if [ ! -f /app/data/pke.sqlite3 ]; then
  echo "No database found at /app/data/pke.sqlite3 — initializing..."
  python -m app.db.init_db
fi

exec "$@"
