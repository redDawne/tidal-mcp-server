#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Tidal MCP — one-click installer for macOS
# Double-click this file in Finder, or run:  bash setup_tidal_mcp.command
#
# It will:
#   1. Copy the server into ~/mcp-servers/tidal_mcp
#   2. Create a Python virtual environment and install dependencies
#   3. Register the server in Claude Desktop's config (without touching your
#      other MCP servers)
#   4. Optionally save your Anthropic API key (for the AI mood-playlist tool)
#   5. Optionally run the one-time Tidal login
# ─────────────────────────────────────────────────────────────────────────────
set -e

DEST="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
VENV_PY="$DEST/.venv/bin/python"

echo ""
echo "=== Tidal MCP setup ==="
echo ""

# 1. Files -----------------------------------------------------------------------
# Installs in place: wherever this script lives is where the server runs from.
echo "→ Installing in place at: $DEST"
if [ ! -f "$DEST/server.py" ]; then
  echo "❌  server.py not found next to this installer. Keep the files together and re-run."
  exit 1
fi

# 2. Virtual env + deps ---------------------------------------------------------
PYBIN="$(command -v python3 || true)"
if [ -z "$PYBIN" ]; then
  echo "❌  python3 not found. Install Python 3 from https://www.python.org/downloads/ and re-run."
  exit 1
fi
echo "→ Creating virtual environment and installing dependencies (this can take a minute)…"
"$PYBIN" -m venv "$DEST/.venv"
"$VENV_PY" -m pip install --quiet --upgrade pip
"$VENV_PY" -m pip install --quiet -r "$DEST/requirements.txt"
echo "✅  Dependencies installed."

# 3. Anthropic API key ----------------------------------------------------------
# The key is never stored in this script. It is taken from the ANTHROPIC_API_KEY
# environment variable if set, otherwise you'll be prompted to paste one.
DEFAULT_ANTHROPIC_KEY="${ANTHROPIC_API_KEY:-}"
echo ""
echo "The AI 'mood playlist' tool uses your Anthropic API key (starts with sk-ant-)."
echo "This is NOT your Tidal login — Tidal is handled separately in step 5."
if [ -n "$DEFAULT_ANTHROPIC_KEY" ]; then
  printf "Press Enter to use the key from your ANTHROPIC_API_KEY environment variable, or paste a different one: "
else
  printf "Paste your Anthropic API key (or press Enter to skip the AI tool): "
fi
read -r ANTHROPIC_KEY
if [ -z "$ANTHROPIC_KEY" ]; then
  ANTHROPIC_KEY="$DEFAULT_ANTHROPIC_KEY"
fi

# 4. Merge into Claude Desktop config ------------------------------------------
echo "→ Registering server in Claude Desktop config…"
mkdir -p "$(dirname "$CONFIG")"
DEST="$DEST" VENV_PY="$VENV_PY" CONFIG="$CONFIG" ANTHROPIC_KEY="$ANTHROPIC_KEY" "$VENV_PY" - <<'PYEOF'
import json, os
config_path = os.environ["CONFIG"]
dest        = os.environ["DEST"]
venv_py     = os.environ["VENV_PY"]
key         = os.environ["ANTHROPIC_KEY"]

data = {}
if os.path.exists(config_path):
    try:
        with open(config_path) as f:
            data = json.load(f)
    except Exception:
        # Back up an unreadable/legacy file rather than destroying it
        os.replace(config_path, config_path + ".bak")
        data = {}

servers = data.setdefault("mcpServers", {})
servers["tidal"] = {
    "command": venv_py,
    "args": [os.path.join(dest, "server.py")],
    "env": {"ANTHROPIC_API_KEY": key},
}
with open(config_path, "w") as f:
    json.dump(data, f, indent=2)
print("   config written:", config_path)
PYEOF
echo "✅  Claude Desktop configured."

# 5. Tidal login (optional) -----------------------------------------------------
echo ""
echo "Tidal itself uses a browser login (no API key). Let's do it now."
printf "Run the one-time Tidal login now? [Y/n]: "
read -r DOLOGIN
case "$DOLOGIN" in
  [Nn]*) echo "→ Skipped. Run later with:";
         echo "     \"$VENV_PY\" \"$DEST/server.py\" --login" ;;
  *)     "$VENV_PY" "$DEST/server.py" --login ;;
esac

echo ""
echo "======================================================================"
echo "Done. Quit and reopen Claude Desktop — the Tidal tools will appear."
echo "======================================================================"
