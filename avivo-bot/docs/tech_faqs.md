# Tech FAQs

## Development Environment Setup

**Q: What OS should I use for development?**
A: Ubuntu 22.04 LTS is the standard. macOS 13+ is also supported. Windows users must use WSL2 with Ubuntu.

**Q: How do I set up my local dev environment?**
A: Clone the monorepo, run `make setup` which installs Python 3.11, Node 20 LTS, Docker Desktop,
and all dependencies. Then copy `.env.example` to `.env` and fill in your credentials.
Contact DevOps on Slack (#devops-help) for secret keys.

**Q: Which Python version is required?**
A: Python 3.11.x. Use `pyenv` to manage versions. Run `pyenv install 3.11.8` then `pyenv local 3.11.8`.

## CI/CD Pipeline

**Q: How does our CI/CD work?**
A: We use GitHub Actions. Every push to a feature branch triggers unit tests + lint.
PRs to `main` trigger integration tests + Docker build. Merges to `main` auto-deploy to staging.
Production deploys require a manual approval gate in GitHub Actions.

**Q: How do I fix a broken pipeline?**
A: Check the Actions tab on GitHub. For Docker build failures, ensure `requirements.txt` is updated.
For test failures, run `pytest` locally first. Ping #ci-cd on Slack if you're stuck.

## Tools & Tech Stack

**Q: What does our core stack look like?**
A: Backend: Python (FastAPI), async everywhere. Data: PostgreSQL 15 + Redis 7.
AI/ML: Ollama (local LLMs), ChromaDB (vector store), sentence-transformers.
Frontend: React + TypeScript. Infra: Docker Compose (dev), Kubernetes (prod).

**Q: What is our code review policy?**
A: All PRs require at least 2 approvals. No self-merging. Reviews must be completed within 2 business days.
Use conventional commits: `feat:`, `fix:`, `chore:`, `docs:`. Squash merge to main.

## Database & Data

**Q: How do we handle DB migrations?**
A: We use Alembic for PostgreSQL. Never modify the DB schema directly. Write a migration:
`alembic revision --autogenerate -m "description"`, review, then `alembic upgrade head`.

**Q: What is our Redis usage policy?**
A: Redis is used for caching (TTL 1hr), session storage (24hr TTL), and message queues (Streams).
Never use Redis as a primary data store. All Redis keys must have a TTL set.

## Debugging & Logging

**Q: How do I enable debug logs?**
A: Set `LOG_LEVEL=DEBUG` in `.env`. Logs are structured JSON via `loguru`. Stream to Grafana Loki in prod.

**Q: What APM tool do we use?**
A: Datadog for prod metrics and traces. Locally use `docker-compose -f docker-compose.monitor.yml up`
to spin up a local Prometheus + Grafana stack.
