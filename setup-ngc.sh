#!/usr/bin/env bash

# Load .env values
set -a
source .env
set +a

if [ -z "$NGC_API_KEY" ]; then
  echo "❌ ERROR: NGC_API_KEY not set in .env"
  exit 1
fi

mkdir -p ~/.docker

cat << EOF > ~/.docker/config.json
{
  "auths": {
    "nvcr.io": {
      "auth": "$(echo -n '$oauthtoken:'$NGC_API_KEY | base64)"
    }
  }
}
EOF

chmod 600 ~/.docker/config.json

echo "✔ NGC authentication configured from .env (nvcr.io ready)"

