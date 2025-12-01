#!/bin/sh
# Simple script to run the list_models.py script using a provided API key or env var

if [ -z "$1" ]; then
  echo "Usage: ./tools/enable_codex.sh <GOOGLE_API_KEY>"
  echo "Or set the GOOGLE_API_KEY environment variable, e.g.: export GOOGLE_API_KEY=KEY"
  python list_models.py
else
  export GOOGLE_API_KEY=$1
  python list_models.py --api-key $1
fi
