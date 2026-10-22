# Publishing to GitHub

The recommended starting point is a **private** GitHub repository, so the contents, authorship, metadata and license can be reviewed before publication. This distribution is not a Git repository until `git init` is executed. You can publish with GitHub CLI (`gh`):

```bash
sudo apt update
sudo apt install git gh

gh auth login             # choose GitHub.com, HTTPS, browser authentication

git init -b main
git add .
git status --short
git commit -m "Initial reproducibility materials for wind reconstruction paper"

gh repo create mass-consistent-wind-reconstruction --private --source=. --remote=origin --push

gh repo view --web
```

If `git commit` reports that your identity is unknown, set `git config user.name "Your Name"` and `git config user.email "you@example.com"` with your real GitHub identity and retry. Do not copy an example email blindly.

**Review before publication**: verify contents, affiliation/author permissions, NASA POWER attribution/version/access dates, any source code licenses, and desired repository visibility. No license has been assigned automatically. Make the repository public only after review:

```bash
gh repo edit --visibility public --accept-visibility-change-consequences
```

This action changes repository visibility; it does not grant rights to third-party material. Confirm the consequences when prompted by GitHub. After publication, capture its URL to add to the manuscript's code/data availability statement and the response to reviewers.

## Alternative (without `gh`)

Create an **empty** repository through GitHub's web UI (do not pre-add a README or license), then from this directory run:

```bash
git init -b main
git add .
git commit -m "Initial reproducibility materials for wind reconstruction paper"
git remote add origin https://github.com/USERNAME/mass-consistent-wind-reconstruction.git
git push -u origin main
```

Replace `USERNAME` with your own account or organization. HTTPS authentication uses GitHub's browser credential flow or a credential helper; do not embed access tokens in remote URLs or source files.
