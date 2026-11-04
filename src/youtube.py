import re
import yt_dlp
from typing import Optional, Dict
from src.logger import logger


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    if not url:
        return ""
    match = re.search(r"(?:v=|\/embed\/|youtu\.be\/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return url.split('=')[-1].split('&')[0]


async def search_youtube_tutorial(topic: str) -> Optional[Dict[str, str]]:
    """
    Search YouTube specifically for practical music production video tutorials.
    Returns a dict with {"url": url, "title": title} or None.
    """
    try:
        # Strip generic question prefixes so search query is laser-targeted
        clean_topic = re.sub(r'^(how\s+to|how\s+do\s+i|what\s+is|explain)\s+', '', topic.strip(), flags=re.IGNORECASE)
        clean_topic = clean_topic.replace('?', '').strip()
        search_query = f"{clean_topic} tutorial music production audio engineering"

        logger.info(f"Searching YouTube tutorial for: {search_query}")

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch1',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
            if result and 'entries' in result and result['entries']:
                entry = result['entries'][0]
                video_id = entry.get('id')
                title = entry.get('title', f"{topic} Tutorial")
                if video_id:
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    logger.info(f"Found YouTube tutorial: '{title}' ({url})")
                    return {"url": url, "title": title}

        return None
    except Exception as e:
        logger.error(f"Error searching YouTube tutorial: {e}")
        return None


async def search_youtube_song(query: str) -> Optional[str]:
    """
    Search for a reference song on YouTube using yt-dlp.
    Returns the URL string.
    """
    try:
        if "remix" not in query.lower() and "audio" not in query.lower():
            search_query = f"{query} audio"
        else:
            search_query = query

        logger.info(f"Searching YouTube for reference song: {search_query}")

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch1',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch1:{search_query}", download=False)
            if result and 'entries' in result and result['entries']:
                video_id = result['entries'][0].get('id')
                if video_id:
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    logger.info(f"Found YouTube reference song: {url}")
                    return url

        return None
    except Exception as e:
        logger.error(f"Error searching YouTube song: {e}")
        return None


def convert_timestamp_to_yt(timestamp: str) -> str:
    """Convert a timestamp string to a URL parameter for YouTube."""
    try:
        if not timestamp or str(timestamp).strip() in ["", "None", "nan"]:
            return ""
        ts_str = str(timestamp).strip()
        if ":" in ts_str:
            parts = ts_str.split(':')
            if len(parts) == 2:
                minutes, seconds = map(int, parts)
                t_seconds = minutes * 60 + seconds
            elif len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                t_seconds = hours * 3600 + minutes * 60 + seconds
            else:
                t_seconds = int(parts[0])
        else:
            t_seconds = int(ts_str)
        return f"&start={t_seconds}"
    except Exception:
        logger.warning(f"Invalid timestamp: {timestamp}")
        return ""


def create_youtube_embed(url: str, timestamp: str = "", label: str = "Video Tutorial", title: str = "") -> str:
    """Create a modern HTML player card with category badge and direct fallback link."""
    video_id = extract_video_id(url)
    if not video_id:
        return ""

    # Clean timestamp parameter
    ts_param = timestamp if str(timestamp).startswith("&") else (f"&start={timestamp}" if timestamp else "")
    embed_url = f"https://www.youtube-nocookie.com/embed/{video_id}?rel=0{ts_param}"
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    if ts_param:
        watch_url += ts_param.replace("&start=", "&t=")

    badge_color = "#3b82f6" if "tutorial" in label.lower() else "#8b5cf6"
    display_title = title if title else "Watch Demonstration"
    if len(display_title) > 55:
        display_title = display_title[:52] + "..."

    html = f"""
<div style="margin: 12px 0; max-width: 580px; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 14px rgba(0,0,0,0.25); background: #16181d; color: #fff; border: 1px solid #2a2e39;">
    <div style="padding: 10px 14px; display: flex; align-items: center; justify-content: space-between; background: #1f232c; border-bottom: 1px solid #2a2e39;">
        <div style="display: flex; align-items: center; gap: 8px; overflow: hidden;">
            <span style="background: {badge_color}; color: #ffffff; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; white-space: nowrap;">
                {label}
            </span>
            <span style="font-size: 13px; font-weight: 600; color: #e2e8f0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{title}">
                {display_title}
            </span>
        </div>
    </div>
    <div style="position: relative; width: 100%; padding-bottom: 56.25%; height: 0; background: #000;">
        <iframe 
            style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;"
            src="{embed_url}" 
            title="YouTube video player" 
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
            referrerpolicy="strict-origin-when-cross-origin" 
            allowfullscreen>
        </iframe>
    </div>
    <div style="padding: 8px 14px; display: flex; justify-content: flex-end; align-items: center; background: #1a1d24;">
        <a href="{watch_url}" target="_blank" rel="noopener noreferrer" 
           style="background: #e11d48; color: white; padding: 5px 12px; border-radius: 6px; text-decoration: none; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; transition: background 0.2s;">
            <span>▶</span> Open in YouTube
        </a>
    </div>
</div>
    """
    return html