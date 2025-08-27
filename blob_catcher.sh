# Put both blob ids here:
BLOBS="11f511cbf622eb2b48f5be91d6b8103c49760a8f"

# Show which commits/paths contain those blobs
for b in $BLOBS; do
  echo "=== Searching for blob $b ==="
  # List every commit and path where the blob appears
  git rev-list --all | while read c; do
    git ls-tree -r "$c" | grep "$b" && echo "in commit $c"
  done
done
