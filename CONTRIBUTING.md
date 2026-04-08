# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.

## Table of Contents

- [Contributing Guidelines](#contributing-guidelines)
  - [Table of Contents](#table-of-contents)
  - [Reporting Bugs/Feature Requests](#reporting-bugsfeature-requests)
  - [Contributing via Pull Requests](#contributing-via-pull-requests)
    - [Best practices](#best-practices)
    - [Getting Started](#getting-started)
      - [Linting/Formatting](#lintingformatting)
      - [Documentation](#documentation)
      - [Testing](#testing)
  - [Finding Contributions to Work On](#finding-contributions-to-work-on)
  - [Code of Conduct](#code-of-conduct)
  - [Security Issue Notifications](#security-issue-notifications)
  - [Licensing](#licensing)

## Reporting Bugs/Feature Requests

We welcome you to use the GitHub issue tracker to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

- A reproducible test case or series of steps
- The version of our code being used
- Any modifications you've made relevant to the bug
- Anything unusual about your environment or deployment

## Contributing via Pull Requests

Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the `main` branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

### Best practices

1. Fork the repository.
2. Commit to your fork using clear commit messages that follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
3. Ensure that linting, formatting and tests are are passing *prior* to raising the pull request.
4. If you are introducing new functionality, please commit the appropriate unit tests.
5. Answer any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.
7. Update `CHANGELOG.md` with any notable changes you make. Be sure to add these changes under `Unreleased`.

### Getting Started

We recommend installing the package locally in editable mode for ease of development.

First, ensure you have [uv](https://docs.astral.sh/uv/) installed. Then, to install the package in editable mode along with the development dependencies and all optional dependencies, run:

```bash
uv sync --all-extras
```

This will install all dependencies (including optional ones like plotting, openai, litellm, and mlflow) and create a virtual environment in `.venv/`.

For a minimal installation with just dev and test dependencies:

```bash
uv sync --all-groups
```

For just dev dependencies without test dependencies:

```bash
uv sync --group dev
```

#### Linting/Formatting

The tools below are used for linting and formatting the codebase.

- [Ruff](https://docs.astral.sh/ruff/)

To check for linting and formatting issues, you can run the following:

```bash
uv run ruff check llmeter/ && uv run ruff format llmeter/
```

Or if not using uv:

```bash
ruff check llmeter/ && ruff format llmeter/
```

#### Documentation

This project uses [Zensical](https://zensical.org/) (A compatible [alernative to MkDocs](https://squidfunk.github.io/mkdocs-material/blog/2025/11/05/zensical/)) to power an interactive docs website, hosted on GitHub pages. This documentation should be updated as part of contributed changes, wherever appropriate.

To work on docs locally, install only the docs dependencies (this avoids pulling in the full project dependency tree):

```bash
uv sync --only-group docs
```

Then preview with the dev server:

```bash
uv run --no-sync zensical serve
```

This starts a preview at http://localhost:8000/ that auto-refreshes as you edit `docs/`.

To do a clean production build:

```bash
uv run --no-sync zensical build --clean
```

⚠️ Note - The following processes are stil **manual** for now (contributions to automate this work are much appreciated!):
- When adding/renaming/deleting .py modules in LLMeter, you need to update the corresponding API reference folders and .md files under `docs/reference`
  - `index.md` files in the API reference should include a top-level title equal to the module name, so they don't just show as "index" in the site. For example, `# callbacks` in [docs/reference/callbacks/index.md](docs/reference/callbacks/index.md).
- The table of contents in [mkdocs.yaml](mkdocs.yaml) should be updated to include all the .md files under `docs/`.

#### Testing

This project uses [pytest](https://docs.pytest.org/en/8.2.x/) for unit testing, which you can invoke using the following:

```bash
uv run pytest
```

Or if not using uv:

```bash
python -m pytest
```

⚠️ Note that integration tests are skipped by default. More details on testing in [tests/README.md](tests/README.md).

## Finding Contributions to Work On

Looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub issue labels, looking for any issues labeled `good first issue` or `help wanted` is a great place to start.

## Code of Conduct

This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
[opensource-codeofconduct@amazon.com](opensource-codeofconduct@amazon.com) with any additional questions or comments.

## Security Issue Notifications

If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.

## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
