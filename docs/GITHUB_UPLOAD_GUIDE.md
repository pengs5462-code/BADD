# GitHub Upload Guide

This guide assumes the remote repository is:

```text
https://github.com/pengs5462-code/BADD
```

## 1. Clone your repository

```bash
git clone https://github.com/pengs5462-code/BADD.git
cd BADD
```

## 2. Copy this revision package into the repository

Assume this package was extracted to `../BADD_github_revision_public/`:

```bash
rsync -av ../BADD_github_revision_public/ ./
```

On Windows PowerShell, if `rsync` is unavailable, use manual copy or:

```powershell
Copy-Item -Recurse -Force ..\BADD_github_revision_public\* .
```

## 3. Append the Git ignore additions

```bash
cat .gitignore_reviewer1_additions >> .gitignore
rm .gitignore_reviewer1_additions
```

PowerShell alternative:

```powershell
Get-Content .gitignore_reviewer1_additions | Add-Content .gitignore
Remove-Item .gitignore_reviewer1_additions
```

## 4. Check that Python files compile

```bash
python -m compileall baddlab tools
```

## 5. Optional one-epoch sanity test

```bash
bash scripts/run_sanity.sh
```

If you are on Windows without Git Bash, run the command inside WSL/Linux server, or manually run the commands inside `scripts/run_sanity.sh`.

## 6. Commit and push

```bash
git status
git add README.md requirements_reviewer1.txt baddlab configs tools scripts docs .gitignore
git commit -m "Add BADD revision reproducibility pipeline"
git push origin main
```

## 7. Verify on GitHub

After pushing, open the repository page and check that the README renders correctly. Confirm that no private files, data folders, checkpoints, or generated experiment outputs were committed.
