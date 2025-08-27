# A. Is the blob currently staged? (index)
git ls-files -s | grep 11f511cbf622eb2b48f5be91d6b8103c49760a8f | awk '{print $4}'

# B. If A shows nothing, look through the current commit(s)
for c in $(git rev-list --all); do
  git ls-tree -r "$c" | grep 11f511cbf622eb2b48f5be91d6b8103c49760a8f && echo "in commit $c"
done

