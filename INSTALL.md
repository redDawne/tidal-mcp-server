# Tidal MCP — Install (macOS)

Everything is prepared. To finish, run the one-click installer.

## Fastest way
1. Open **Finder → Downloads → tidal_mcp**.
2. Double-click **`setup_tidal_mcp.command`**.
   - If macOS blocks it ("unidentified developer"), right-click → **Open** → **Open**.
3. Follow the prompts. It will:
   - Install the server to `~/mcp-servers/tidal_mcp`
   - Create a Python environment and install dependencies
   - Register the server in Claude Desktop (your other MCP servers are left untouched)
   - Ask for your **Anthropic API key** (for the AI mood-playlist tool — optional, you can skip)
   - Offer to run the **one-time Tidal login** (opens a link in your browser)
4. **Quit and reopen Claude Desktop.** The Tidal tools appear automatically.

## If you prefer Terminal
```bash
bash ~/Downloads/tidal_mcp/setup_tidal_mcp.command
```

## Notes
- **Two different credentials:**
  - *Tidal* = browser login (no key). Handled in the installer's last step.
  - *Anthropic key* (`sk-ant-…`) = only needed for the `tidal_mood_playlist` AI tool. Everything else (search, playlists, etc.) works without it.
- To re-run the Tidal login later:
  `~/mcp-servers/tidal_mcp/.venv/bin/python ~/mcp-servers/tidal_mcp/server.py --login`
- To add/replace your Anthropic key later, edit the `ANTHROPIC_API_KEY` value in:
  `~/Library/Application Support/Claude/claude_desktop_config.json`

## Available tools once installed
Search tracks/artists, get artist top tracks, list/read/create/delete playlists,
add tracks to a playlist, and AI mood-based playlist generation.
