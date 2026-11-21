#!/bin/sh
# Container start: fetch + prepare the model into the /data/models volume once, then serve on 0.0.0.0:8080.
set -e
mkdir -p /data/models
ln -sfn /data/models /opt/openset-llm/models
[ -f /data/models/model.gguf ] || /opt/openset-llm/app/install.sh --model-only
exec /opt/openset-llm/app/serve --host 0.0.0.0 --port 8080 "$@"
