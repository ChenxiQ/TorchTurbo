# Contribution Guide
Our goal is to build an open, inclusive and collaborative open source community. Everyone is welcome to participate.

## Report Bugs
Ensure the bug was not already reported by searching on Github Issues. If you cannot find an open issue addressing the problem, feel free to create a new one.

## Propose Improvements or New Features
Submit your proposal on Github Issues. Ensure your proposal includes sufficient details to clearly convey your idea.

If you plan to implement your proposal to contribute to the community, please follow the following sections.

## Contribute Code
Welcome all kinds of code contributions including bug fixes, new features or improvements to existing code (e.g., performance, accessibility, readability, etc).

Not sure where to start? Browse Github Issues, where you may find interesting topics to work on. For new contributors, it's a good idea to start with issues labeled as "good first issue".

### 1. Make code changes
Before starting, indicate your interest in working on a specific issue by commenting on it or creating a new one. It is best to have a full discussion with community members before working on a challenging or large-scale issue.

Follow these steps:

1. Fork the repository and clone your fork, adding the base repo as a remote:

   ```bash
   git clone git@github.com:<your-handle>/TorchTurbo.git
   cd TorchTurbo
   git remote add upstream https://github.com/inclusionAI/TorchTurbo.git
   # Pull from upstream, but never push to it — push only to your fork (origin)
   git remote set-url --push upstream no-pushing
   ```

2. Create a branch for your changes — **do not** work on `main`:

   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git checkout -b a-descriptive-name-for-my-changes
   ```

3. Set up a development install

4. Check before commit:

   Commit messages must use one of these prefixes:
   `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`,
   `chore`, `revert`. For example:

   ```
   feat: add HSDP support
   fix: correct xxx OP precision
   docs: update params for HSDP
   ```

5. Commit with a clear message, keep your branch synced, and push:

   ```bash
   git fetch upstream
   git rebase upstream/main
   git push -u origin a-descriptive-name-for-my-changes
   ```

### 2. Create a pull request
Open the pull request from your fork. The title of a pull request must begin with one of the following labels to indicate the type of the pull request: fix, feat, docs, style, refactor, enhance, test, build, ci, chore, or revert.

### 3. Get a code review
TorchTurbo Maintainers will regularly check for unreviewed pull requests. If a pull request has received more than one approval from Maintainers and all "Request Changes" from Committers/Maintainers are resolved, it can be merged by any Maintainer.

If your pull request has not received a response for days, you can find potential reviewers from Maintainers list and get their attention by tagging them using @name in a comment.