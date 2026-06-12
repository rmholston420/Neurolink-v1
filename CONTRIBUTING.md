# Contributing to Neurolink-v1

## First-Time Setup

### 1. Clone and install

```bash
git clone https://github.com/rmholston420/Neurolink-v1.git
cd Neurolink-v1
```

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**Frontend:**
```bash
cd frontend
npm install
```

### 2. Untrack committed node_modules (one-time)

`frontend/node_modules/` was inadvertently committed to the git index before
the `.gitignore` rule took effect. Run this once to clean it up:

```bash
# From repo root:
git rm -r --cached frontend/node_modules/
git commit -m "chore: untrack frontend/node_modules from index"
git push
```

Or use the helper script:

```bash
chmod +x scripts/untrack-node-modules.sh
./scripts/untrack-node-modules.sh
git commit -m "chore: untrack frontend/node_modules from index"
git push
```

---

## Running Tests

### Backend (pytest)

```bash
cd backend
pytest --cov=src/neurolink --cov-report=term-missing
```

Linting:
```bash
ruff check .
mypy src/neurolink/
```

### Frontend (vitest)

```bash
cd frontend
npm test                  # single pass
npm run test:watch        # watch mode
npm run test:coverage     # with v8 coverage report
```

---

## Branch / PR Workflow

1. Branch from `main`: `git checkout -b feat/your-feature`
2. Keep commits atomic and conventionally named (`feat:`, `fix:`, `chore:`, `test:`)
3. Ensure `ruff check .` and `mypy src/neurolink/` are clean before opening a PR
4. Ensure `npm test` passes with no failures
5. Open a PR against `main`
