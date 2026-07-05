# tidal_mcp — Tidal MCP Server for Claude

Gives Claude tools to search, browse, and create playlists on Tidal,
including AI-powered mood-based playlist generation.

## Setup

### 1. Install dependencies
```bash
pip install tidalapi anthropic "mcp[cli]"
```

### 2. Authenticate with Tidal (one-time)
```bash
python server.py --login
```
Open the link that appears, approve access. Session is saved to
`~/.tidal_mcp_session.json` and reused automatically.

### 3. Configure Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tidal": {
      "command": "python",
      "args": ["/absolute/path/to/tidal_mcp/server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-your-key-here"
      }
    }
  }
}
```

Restart Claude Desktop. The Tidal tools will appear automatically.

## Available Tools

| Tool | What it does |
|------|-------------|
| `tidal_search_tracks` | Search the Tidal catalog by keyword |
| `tidal_search_artists` | Find artists by name |
| `tidal_get_artist_top_tracks` | Get an artist's most popular tracks |
| `tidal_list_my_playlists` | List your Tidal playlists |
| `tidal_get_playlist_tracks` | See what's in a playlist |
| `tidal_create_playlist` | Create a playlist with specific track IDs |
| `tidal_add_tracks_to_playlist` | Add tracks to an existing playlist |
| `tidal_delete_playlist` | Delete a playlist |
| `tidal_mood_playlist` | **AI-powered** — describe a mood, get a playlist |

## Example Prompts in Claude

- *"Make me a rainy Sunday morning playlist — melancholic, piano-forward, not too slow"*
- *"Show me my Tidal playlists"*
- *"Search Tidal for dark ambient tracks"*
- *"What are Nils Frahm's top tracks on Tidal?"*
- *"Do a dry run of a late night jazz playlist so I can see the tracklist first"*
- *"Add those tracks to my existing Evening Wind-Down playlist"*
