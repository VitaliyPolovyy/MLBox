# MLBox Constitution

## Core Principles

### I. Fail-Fast Error Handling (NON-NEGOTIABLE)

* Functions focus on pure business logic; errors bubble up to orchestration level
* No defensive try-catch blocks in individual functions
* Error handling is a separate concern from business logic

### II. Ray Serve Services

* All ML services are Ray Serve deployments with clear service boundaries
* HTTP/REST interfaces with JSON schema validation
* Each service independently deployable, testable, and maintainable

### III. Poetry Dependencies

* All dependencies managed via Poetry
* `pyproject.toml` is single source of truth
* No `pip install` commands in production code or deployment scripts

### IV. Library and Framework Approval

* New libraries and frameworks require approval before adding
* Justify why existing solutions are insufficient
* Evaluate impact on Docker image size and dependencies

### V. Code Quality Standards

* Code must pass `black` formatting and `pylint` checks before commit
* Follow existing code patterns and conventions

### VI. Security

* All secrets, tokens, and API keys stored in environment variables
* Never hardcode credentials in code
* Use `.env` files for local development (excluded from git)

### VII. Docker Image Size

* Prefer lightweight libraries over heavyweight when functionality is equivalent
* Evaluate image size impact before adding dependencies

### VIII. Python 3.11

* Strictly enforced via `requires-python` in pyproject.toml

### IX. Git Commits

* One commit per completed task
* Commit messages should clearly describe what was done

## Governance

* Constitution supersedes all other development practices
* Amendments require documentation and approval
* All PRs must verify compliance with these principles
* Deviations require explicit documentation and approval

**Version**: 2.2.0 | **Ratified**: 2025-11-25 | **Last Amended**: 2025-11-25
