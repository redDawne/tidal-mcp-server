#!/usr/bin/env python3
"""
tidal_mcp — MCP server for Tidal music platform
Provides Claude with tools to search, browse, and create playlists on Tidal,
with AI-powered mood-based playlist generation via the Anthropic API.

Requirements:
    pip install tidalapi anthropic mcp[cli]

Run:
    python server.py

Configure in Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "tidal": {
          "command": "python",
          "args": ["/path/to/tidal_mcp/server.py"],
          "env": {
            "ANTHROPIC_API_KEY": "sk-ant-..."
          }
        }
      }
    }
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import anthropic
import tidalapi
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP, Context


# ── Constants ──────────────────────────────────────────────────────────────────

TIDAL_SESSION_FILE = Path.home() / ".tidal_mcp_session.json"
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL       = "claude-sonnet-4-20250514"

DEFAULT_SEARCH_LIMIT   = 20
DEFAULT_PLAYLIST_SIZE  = 20
SEARCH_PER_QUERY       = 8


# ── Lifespan: shared Tidal session ────────────────────────────────────────────

@asynccontextmanager
async def app_lifespan(server):
    """Initialize and share a single authenticated Tidal session."""
    session = tidalapi.Session()
    loaded = False

    if os.path.exists(TIDAL_SESSION_FILE):
        try:
            session.load_session_from_file(TIDAL_SESSION_FILE)
            if session.check_login():
                loaded = True
        except Exception:
            pass

    if not loaded:
        # For a local stdio MCP server, print to stderr so Claude Desktop
        # can show the link to the user in its log output.
        print(
            "\n[tidal_mcp] Tidal login required.\n"
            "[tidal_mcp] Run `python server.py --login` first to authenticate.\n",
            file=sys.stderr,
        )

    yield {"tidal": session, "authenticated": loaded}


mcp = FastMCP("tidal_mcp", lifespan=app_lifespan)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_session(ctx) -> tidalapi.Session:
    """Return the shared Tidal session, raising if not authenticated."""
    state = ctx.request_context.lifespan_context
    if not state.get("authenticated"):
        raise RuntimeError(
            "Not logged in to Tidal. Run `python server.py --login` in a terminal first."
        )
    return state["tidal"]


def _track_to_dict(track: tidalapi.Track) -> dict:
    return {
        "id": track.id,
        "title": track.name,
        "artist": track.artist.name if track.artist else "Unknown",
        "album": track.album.name if track.album else "Unknown",
        "duration_s": track.duration,
        "duration": f"{track.duration // 60}:{track.duration % 60:02d}",
    }


def _playlist_to_dict(pl) -> dict:
    return {
        "id": str(pl.id),
        "name": pl.name,
        "description": pl.description or "",
        "num_tracks": pl.num_tracks,
        "url": pl.listen_url,
    }


def _handle_error(e: Exception) -> str:
    if isinstance(e, RuntimeError):
        return f"Error: {e}"
    return f"Error: {type(e).__name__}: {e}"


# ── Input Models ───────────────────────────────────────────────────────────────

class SearchTracksInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Search query (artist, title, album, genre)", min_length=1, max_length=200)
    limit: Optional[int] = Field(default=DEFAULT_SEARCH_LIMIT, description="Max results to return", ge=1, le=50)


class SearchArtistsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Artist name to search for", min_length=1, max_length=200)
    limit: Optional[int] = Field(default=10, ge=1, le=20)


class GetArtistTopTracksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artist_id: int = Field(..., description="Tidal artist ID (from tidal_search_artists)")
    limit: Optional[int] = Field(default=10, ge=1, le=30)


class ListMyPlaylistsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=20, ge=1, le=50)
    offset: Optional[int] = Field(default=0, ge=0)


class GetPlaylistTracksInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    playlist_id: str = Field(..., description="Tidal playlist UUID (from tidal_list_my_playlists)", min_length=1)
    limit: Optional[int] = Field(default=50, ge=1, le=100)


class CreatePlaylistInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., description="Name for the new playlist", min_length=1, max_length=200)
    description: Optional[str] = Field(default="", description="Optional description", max_length=500)
    track_ids: list[int] = Field(..., description="List of Tidal track IDs to add", min_length=1, max_length=500)


class MoodPlaylistInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    mood: str = Field(
        ...,
        description="Natural language mood/vibe description, e.g. 'rainy Sunday morning, melancholic piano, not too slow'",
        min_length=3,
        max_length=500,
    )
    playlist_size: Optional[int] = Field(default=DEFAULT_PLAYLIST_SIZE, ge=5, le=50)
    dry_run: Optional[bool] = Field(
        default=False,
        description="If true, return the planned tracklist without creating the playlist in Tidal",
    )


class AddTracksToPlaylistInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    playlist_id: str = Field(..., description="UUID of the Tidal playlist to add tracks to", min_length=1)
    track_ids: list[int] = Field(..., description="List of Tidal track IDs to add", min_length=1, max_length=200)


class DeletePlaylistInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    playlist_id: str = Field(..., description="UUID of the Tidal playlist to delete", min_length=1)


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="tidal_search_tracks",
    annotations={
        "title": "Search Tidal for Tracks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tidal_search_tracks(params: SearchTracksInput, ctx: Context) -> str:
    """Search the Tidal catalog for tracks matching a query.

    Args:
        params.query: Search string — artist name, song title, album, genre, mood keyword
        params.limit: Max number of tracks to return (default 20)

    Returns:
        JSON list of tracks with id, title, artist, album, duration
    """
    try:
        session = _get_session(ctx)
        results = session.search(params.query, models=[tidalapi.Track], limit=params.limit)
        tracks = [_track_to_dict(t) for t in (results.get("tracks") or [])]
        return json.dumps({"query": params.query, "count": len(tracks), "tracks": tracks}, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="tidal_search_artists",
    annotations={
        "title": "Search Tidal for Artists",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tidal_search_artists(params: SearchArtistsInput, ctx: Context) -> str:
    """Search the Tidal catalog for artists by name.

    Args:
        params.query: Artist name to search
        params.limit: Max results (default 10)

    Returns:
        JSON list of artists with id, name, popularity
    """
    try:
        session = _get_session(ctx)
        results = session.search(params.query, models=[tidalapi.Artist], limit=params.limit)
        artists = [
            {"id": a.id, "name": a.name}
            for a in (results.get("artists") or [])
        ]
        return json.dumps({"query": params.query, "count": len(artists), "artists": artists}, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="tidal_get_artist_top_tracks",
    annotations={
        "title": "Get Artist's Top Tracks on Tidal",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def tidal_get_artist_top_tracks(params: GetArtistTopTracksInput, ctx: Context) -> str:
    """Get the top tracks for a specific Tidal artist.

    Args:
        params.artist_id: Tidal numeric artist ID (from tidal_search_artists)
        params.limit: Number of top tracks to return (default 10)

    Returns:
        JSON list of tracks with id, title, artist, album, duration
    """
    try:
        session = _get_session(ctx)
        artist = session.artist(params.artist_id)
        tracks = [_track_to_dict(t) for t in artist.get_top_tracks(limit=params.limit)]
        return json.dumps({"artist_id": params.artist_id, "artist": artist.name, "tracks": tracks}, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="tidal_list_my_playlists",
    annotations={
        "title": "List My Tidal Playlists",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tidal_list_my_playlists(params: ListMyPlaylistsInput, ctx: Context) -> str:
    """List the authenticated user's Tidal playlists.

    Args:
        params.limit: Max playlists to return (default 20)
        params.offset: Pagination offset (default 0)

    Returns:
        JSON list of playlists with id, name, description, num_tracks, url
    """
    try:
        session = _get_session(ctx)
        all_playlists = session.user.playlists()
        page = all_playlists[params.offset: params.offset + params.limit]
        playlists = [_playlist_to_dict(pl) for pl in page]
        return json.dumps({
            "count": len(playlists),
            "total": len(all_playlists),
            "offset": params.offset,
            "has_more": params.offset + params.limit < len(all_playlists),
            "playlists": playlists,
        }, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="tidal_get_playlist_tracks",
    annotations={
        "title": "Get Tracks in a Tidal Playlist",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
)
async def tidal_get_playlist_tracks(params: GetPlaylistTracksInput, ctx: Context) -> str:
    """Retrieve the tracks in a specific Tidal playlist.

    Args:
        params.playlist_id: UUID of the playlist (from tidal_list_my_playlists)
        params.limit: Max tracks to return (default 50)

    Returns:
        JSON list of tracks with id, title, artist, album, duration
    """
    try:
        session = _get_session(ctx)
        playlist = session.playlist(params.playlist_id)
        tracks = [_track_to_dict(t) for t in playlist.tracks(limit=params.limit)]
        return json.dumps({
            "playlist": params.playlist_id,
            "name": playlist.name,
            "count": len(tracks),
            "tracks": tracks,
        }, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="tidal_create_playlist",
    annotations={
        "title": "Create a Tidal Playlist",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def tidal_create_playlist(params: CreatePlaylistInput, ctx: Context) -> str:
    """Create a new playlist in the user's Tidal account and populate it with tracks.

    Args:
        params.name: Playlist name
        params.description: Optional playlist description
        params.track_ids: List of Tidal track IDs to add

    Returns:
        JSON with the new playlist id, name, url
    """
    try:
        session = _get_session(ctx)
        playlist = session.user.create_playlist(params.name, params.description or "")
        playlist.add(params.track_ids)
        return json.dumps({
            "status": "created",
            "playlist": _playlist_to_dict(playlist),
        }, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="tidal_add_tracks_to_playlist",
    annotations={
        "title": "Add Tracks to an Existing Tidal Playlist",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def tidal_add_tracks_to_playlist(params: AddTracksToPlaylistInput, ctx: Context) -> str:
    """Add tracks to an existing Tidal playlist.

    Args:
        params.playlist_id: UUID of the target playlist
        params.track_ids: List of Tidal track IDs to add

    Returns:
        JSON confirmation with updated track count
    """
    try:
        session = _get_session(ctx)
        playlist = session.playlist(params.playlist_id)
        if not isinstance(playlist, tidalapi.UserPlaylist):
            return "Error: This is not a user-owned playlist and cannot be modified."
        playlist.add(params.track_ids)
        return json.dumps({
            "status": "tracks_added",
            "playlist_id": params.playlist_id,
            "tracks_added": len(params.track_ids),
            "total_tracks": playlist.num_tracks,
        }, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="tidal_delete_playlist",
    annotations={
        "title": "Delete a Tidal Playlist",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False,
    },
)
async def tidal_delete_playlist(params: DeletePlaylistInput, ctx: Context) -> str:
    """Permanently delete a Tidal playlist from the user's account.

    Args:
        params.playlist_id: UUID of the playlist to delete

    Returns:
        JSON confirmation of deletion
    """
    try:
        session = _get_session(ctx)
        playlist = session.playlist(params.playlist_id)
        name = playlist.name
        playlist.delete()
        return json.dumps({"status": "deleted", "playlist_id": params.playlist_id, "name": name}, indent=2)
    except Exception as e:
        return _handle_error(e)


@mcp.tool(
    name="tidal_mood_playlist",
    annotations={
        "title": "Generate a Mood-Based Tidal Playlist with AI",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def tidal_mood_playlist(params: MoodPlaylistInput, ctx: Context) -> str:
    """Generate a Tidal playlist from a natural language mood or vibe description.

    Uses Claude AI to interpret the mood, build Tidal search queries, curate
    the best matching tracks, and create the playlist in the user's account.

    Args:
        params.mood: Natural language description, e.g. 'late night drive, cinematic, melancholic'
        params.playlist_size: Number of tracks to include (default 20)
        params.dry_run: If true, return the planned tracklist without creating in Tidal

    Returns:
        JSON with playlist details and tracklist (or dry-run preview)
    """
    if not ANTHROPIC_API_KEY:
        return "Error: ANTHROPIC_API_KEY environment variable not set. Cannot use AI playlist generation."

    try:
        session = _get_session(ctx)
        claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # ── Step 1: Claude generates a search strategy ──────────────────────
        strategy_prompt = """\
You are a music curator specializing in Tidal playlists.
Given a mood/vibe, return ONLY valid JSON (no markdown) in this shape:
{
  "playlist_name": "<evocative name>",
  "playlist_description": "<1-2 sentences>",
  "search_queries": ["<q1>", "<q2>", ...],
  "artists_to_include": ["<artist1>", ...],
  "avoid_keywords": ["<word1>", ...]
}
Rules:
- search_queries: 4-6 short Tidal search strings (genre + mood + tempo cues)
- artists_to_include: 3-6 artists that fit the vibe
- avoid_keywords: words signalling a bad fit; empty [] if none
"""
        strategy_resp = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=800,
            system=strategy_prompt,
            messages=[{"role": "user", "content": params.mood}],
        )
        raw_strategy = strategy_resp.content[0].text.strip()
        try:
            strategy = json.loads(raw_strategy)
        except json.JSONDecodeError:
            import re
            m = re.search(r'\{.*\}', raw_strategy, re.DOTALL)
            strategy = json.loads(m.group()) if m else {}
            if not strategy:
                return f"Error: Could not parse AI strategy response: {raw_strategy[:200]}"

        # ── Step 2: Search Tidal ─────────────────────────────────────────────
        seen_ids: set[int] = set()
        candidates: list[dict] = []

        def add_track(track: tidalapi.Track) -> None:
            if track.id not in seen_ids:
                seen_ids.add(track.id)
                candidates.append(_track_to_dict(track))

        for query in strategy.get("search_queries", []):
            try:
                results = session.search(query, models=[tidalapi.Track], limit=SEARCH_PER_QUERY)
                for t in (results.get("tracks") or []):
                    add_track(t)
            except Exception:
                pass

        for artist_name in strategy.get("artists_to_include", []):
            try:
                results = session.search(artist_name, models=[tidalapi.Artist], limit=3)
                artist_list = results.get("artists") or []
                if artist_list:
                    for t in artist_list[0].get_top_tracks(limit=5):
                        add_track(t)
            except Exception:
                pass

        if not candidates:
            return "Error: No tracks found for this mood. Try a different description."

        # ── Step 3: Claude curates the tracklist ─────────────────────────────
        curation_system = f"""\
You are a music curator. Select the best {params.playlist_size} tracks from the
candidate pool for the given mood.

Return ONLY a JSON array of integer track IDs ordered for best listening flow.
Prioritise: mood cohesion, artist variety (max 2 per artist), good energy arc.
Avoid tracks whose titles contain the avoid_keywords.
"""
        curation_user = (
            f"Mood: {params.mood}\n"
            f"Avoid keywords: {strategy.get('avoid_keywords', [])}\n\n"
            f"Candidates:\n"
            + json.dumps(
                [{"id": t["id"], "title": t["title"], "artist": t["artist"], "duration_s": t["duration_s"]}
                 for t in candidates],
                indent=2,
            )
        )
        curation_resp = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=400,
            system=curation_system,
            messages=[{"role": "user", "content": curation_user}],
        )
        raw_ids = curation_resp.content[0].text.strip()
        try:
            selected_ids = [int(i) for i in json.loads(raw_ids)]
        except (json.JSONDecodeError, ValueError):
            selected_ids = [t["id"] for t in candidates[: params.playlist_size]]

        # Build the ordered track list for the response
        id_to_track = {t["id"]: t for t in candidates}
        selected_tracks = [id_to_track[tid] for tid in selected_ids if tid in id_to_track]

        # ── Step 4: Create or dry-run ─────────────────────────────────────────
        if params.dry_run:
            total_s = sum(t["duration_s"] for t in selected_tracks)
            return json.dumps({
                "dry_run": True,
                "playlist_name": strategy["playlist_name"],
                "playlist_description": strategy["playlist_description"],
                "search_queries": strategy["search_queries"],
                "artists_targeted": strategy["artists_to_include"],
                "total_runtime_min": total_s // 60,
                "track_count": len(selected_tracks),
                "tracks": selected_tracks,
            }, indent=2)

        playlist = session.user.create_playlist(
            strategy["playlist_name"],
            strategy["playlist_description"],
        )
        playlist.add(selected_ids)

        total_s = sum(t["duration_s"] for t in selected_tracks)
        return json.dumps({
            "status": "created",
            "playlist": _playlist_to_dict(playlist),
            "total_runtime_min": total_s // 60,
            "track_count": len(selected_tracks),
            "tracks": selected_tracks,
        }, indent=2)

    except Exception as e:
        return _handle_error(e)


# ── Login helper (run directly) ───────────────────────────────────────────────

def do_login() -> None:
    """Standalone login flow — run once before starting the MCP server."""
    print("Tidal MCP — initial login")
    session = tidalapi.Session()
    session.login_oauth_simple()
    if session.check_login():
        session.save_session_to_file(TIDAL_SESSION_FILE)
        print(f"✅  Logged in. Session saved to {TIDAL_SESSION_FILE}")
        print("    You can now start the MCP server normally.")
    else:
        print("❌  Login failed.")
        sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--login":
        do_login()
    else:
        mcp.run()
