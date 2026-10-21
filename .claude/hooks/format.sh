#!/bin/bash
# Format only the file that was just edited (formatting the whole repo creates unrelated diffs).
file=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("file_path", ""))')
case "$file" in
  *.py) cd "$CLAUDE_PROJECT_DIR" && uv run --no-sync ruff format --quiet "$file" && uv run --no-sync ruff check --fix --quiet "$file" ;;
esac
exit 0
