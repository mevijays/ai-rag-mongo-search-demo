#!/usr/bin/env bash
set -euo pipefail

# Load env vars
if [ -f .env ]; then
  set -a
  source .env
  set +a
else
  echo ".env not found. Please create it first."
  exit 1
fi

echo "Starting MongoDB with docker compose..."
docker compose up -d mongodb

echo "Waiting for MongoDB to be ready..."
retries=120
until docker exec rag-mongodb mongo -u "$MONGO_INITDB_ROOT_USERNAME" -p "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin --quiet --eval 'db.runCommand({ ping: 1 })' >/dev/null 2>&1; do
  ((retries--)) || { echo "MongoDB did not become ready in time"; exit 1; }
  sleep 2
done

echo "MongoDB responded to ping. Double-checking before seeding..."
sleep 2

echo "Seeding sample documents into MongoDB (no embeddings)..."
seed_retries=5
until docker exec -i rag-mongodb mongo \
  -u "$MONGO_INITDB_ROOT_USERNAME" \
  -p "$MONGO_INITDB_ROOT_PASSWORD" \
  --authenticationDatabase admin <<'MONGO_SEED'
use ragdb
db.documents.insertMany([
  { title: "Welcome", source: "seed", content: "This is a tiny sample document about our RAG demo. It explains how the system works and how to search.", chunk_index: 0, created_at: new Date() },
  { title: "FAQ", source: "seed", content: "Frequently asked questions include: how to upload files and how to query the knowledge base.", chunk_index: 0, created_at: new Date() }
], { ordered: false })
MONGO_SEED
do
  ((seed_retries--)) || { echo "Failed to seed sample documents after multiple attempts"; exit 1; }
  echo "Seed failed, retrying in 3s..."
  sleep 3
done

echo "Done. MongoDB is up and seeded."
echo "Next steps: create and activate a Python env, install requirements, and run app.py"