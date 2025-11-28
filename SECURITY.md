# Security Guidance

If you see an API key or other secret in this repository, follow these steps immediately:

1. Rotate the compromised credential (e.g., regenerate a new Google API key).
2. Remove the secret from the repository's history using a tool like `git filter-repo` or `BFG`.
3. Add the file path to `.gitignore` to prevent re-adding secrets accidentally.
4. Add a `backend/.env.example` and replace any committed `.env` with the example.
5. Consider adding a commit hook or pre-commit hook to detect secrets (`git-secrets` or `pre-commit`).

Example to remove a key with `git filter-repo`:

```bash
git filter-repo --path backend/.env --invert-paths
```

After rewriting history, force-push the cleaned branch and invalidate old keys. This will require coordination with team members as it rewrites history.
