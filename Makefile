# agent-hotwash -- uv-based dev tasks. Run `make` or `make help` for the list.
# Convention: a `## description` after a target is picked up by the help target.

# Shared lint/format/typecheck target set.
SRC := src tests

# Tool invocations live here once so the verbose targets and the quiet `check`
# gate run byte-identical commands.
PYTEST        := uv run pytest
TEST_CMD      := $(PYTEST)
LINT_CMD      := uv run ruff check $(SRC)
FMT_CHECK_CMD := uv run ruff format --check $(SRC)
TYPECHECK_CMD := uv run ty check --force-exclude $(SRC)

# ANSI colors for the check summary (export NO_COLOR=1 to disable).
ifndef NO_COLOR
GREEN := \033[32m
RED   := \033[31m
BOLD  := \033[1m
RESET := \033[0m
endif

# run_quiet: run a command with NO output on success (success = silence, the
# Unix way) and the full combined stdout+stderr on failure. Used by the
# standalone static-check targets (`make lint`, `make typecheck`, ...).
#   usage:  @$(call run_quiet,$(LINT_CMD))
define run_quiet
out=$$($(1) 2>&1); st=$$?; [ $$st -eq 0 ] || { printf '%s\n' "$$out"; exit $$st; }
endef

# run_check: like run_quiet but prints a one-line "<label>: OK" on success (and
# "<label>: FAIL" + the full output on failure). Used by the `make check` gate
# so each leg reports that it ran. Output is one atomic block, so it stays
# readable even when several run_check recipes run concurrently under `make -j`.
#   usage:  @$(call run_check,Lint,$(LINT_CMD))
define run_check
out=$$($(2) 2>&1); st=$$?; if [ $$st -eq 0 ]; then printf '  $(GREEN)%-14s OK$(RESET)\n' '$(1):'; else printf '  $(RED)%-14s FAIL$(RESET)\n' '$(1):'; printf '%s\n' "$$out"; exit $$st; fi
endef

.PHONY: help install sync format fmt lint format-check typecheck test check \
	_ck-fmt _ck-lint _ck-type _ck-test release clean

help: ## Show this help (default target).
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Sync the virtualenv from uv.lock (incl. dev group).
	uv sync

sync: install ## Alias for install.

format: ## Format src + tests with ruff (mutates files).
	uv run ruff format $(SRC)

fmt: format ## Alias for format.

# The standalone static-check targets are SILENT on success and dump the full
# combined stdout+stderr only when the tool exits non-zero. (`make check`
# reports each as "<name>: OK" via its own wrappers below.)
lint: ## Lint src + tests with ruff (silent unless it fails).
	@$(call run_quiet,$(LINT_CMD))

format-check: ## Check formatting with ruff (silent unless it fails).
	@$(call run_quiet,$(FMT_CHECK_CMD))

typecheck: ## Type-check src + tests with ty (silent unless it fails).
	@$(call run_quiet,$(TYPECHECK_CMD))

test: ## Run the test suite (verbose).
	$(TEST_CMD)

# check runs every leg via `make -j` so the static checks and tests run
# concurrently. Output is captured per target, so the summary lines never
# interleave. Quiet on success: each leg prints one "<name>: OK" line.
check: ## Full hygiene gate: format/lint/typecheck + tests (parallel). Quiet on success.
	@printf '$(BOLD)Running make check...$(RESET)\n'
	@$(MAKE) -j --no-print-directory _ck-fmt _ck-lint _ck-type _ck-test
	@printf '$(GREEN)$(BOLD)All checks passed.$(RESET)\n'

_ck-fmt:
	@$(call run_check,Format,$(FMT_CHECK_CMD))

_ck-lint:
	@$(call run_check,Lint,$(LINT_CMD))

_ck-type:
	@$(call run_check,Typecheck,$(TYPECHECK_CMD))

_ck-test:
	@$(call run_check,Tests,$(TEST_CMD))

# release: bump the version, tag it, and build the wheel/sdist. Pass the bump
# level via BUMP (major|minor|patch; default patch), e.g. `make release BUMP=minor`.
# Runs the full check gate first so a broken tree never gets tagged.
BUMP ?= patch
release: check ## Bump version (BUMP=major|minor|patch), git-tag, and build dists.
	@[ -z "$$(git status --porcelain)" ] || { echo "Release: FAIL (working tree not clean)"; exit 1; }
	uv version --bump $(BUMP)
	@version=$$(uv version --short); \
		git commit -am "release: v$$version" && \
		git tag "v$$version" && \
		printf '$(GREEN)Tagged v%s$(RESET)\n' "$$version"
	uv build

clean: ## Remove caches and build artifacts (keeps .venv and uv.lock).
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ruff_cache .pytest_cache dist build
	find . -type f -name '*.pyc' -delete
