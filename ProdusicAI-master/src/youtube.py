import re
import yt_dlp
from src.logger import logger


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    if not url:
        return ""
    match = re.search(r"(?:v=|\/embed\/|youtu\.be\/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return url.split('=')[-1].split('&')[0]


async def search_youtube_song(query: str) -> str:
    """
    Search for a song on YouTube using yt-dlp
    """
    try:
        if "remix" not in query.lower():
            query = f"{query} official"

        logger.info(f"Searching YouTube for: {query}")

        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch1',
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if result and 'entries' in result and result['entries']:
                video_id = result['entries'][0].get('id')
                if video_id:
                    url = f"https://www.youtube.com/watch?v={video_id}"
                    logger.info(f"Found YouTube video: {url}")
                    return url

        logger.warning(f"No YouTube videos found for: {query}")
        return None

    except Exception as e:
        logger.error(f"Error searching YouTube: {str(e)}")
        return None


def convert_timestamp_to_yt(timestamp: str) -> str:
    """Convert a timestamp string to a URL parameter for YouTube."""
    try:
        if not timestamp or str(timestamp).strip() in ["", "None", "nan"]:
            return ""
        ts_str = str(timestamp).strip()
        if ":" in ts_str:
            minutes, seconds = map(int, ts_str.split(':'))
            t_seconds = minutes * 60 + seconds
        else:
            t_seconds = int(ts_str)
        return f"&start={t_seconds}"
    except Exception:
        logger.warning(f"Invalid timestamp: {timestamp}")
        return ""


def create_youtube_embed(url: str, timestamp: str = "") -> str:
    """Create an HTML iframe and direct fallback link for a YouTube URL."""
    video_id = extract_video_id(url)
    if not video_id:
        return ""

    # Clean timestamp parameter
    ts_param = timestamp if str(timestamp).startswith("&") else (f"&start={timestamp}" if timestamp else "")
    embed_url = f"https://www.youtube-nocookie.com/embed/{video_id}?rel=0{ts_param}"
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    if ts_param:
        watch_url += ts_param.replace("&start=", "&t=")

    html = f"""
<div style="margin: 12px 0; max-width: 560px; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.15); background: #181818; color: #fff;">
    <iframe 
        width="100%" height="315" 
        src="{embed_url}" 
        title="YouTube video player" frameborder="0" 
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
        referrerpolicy="strict-origin-when-cross-origin" 
        allowfullscreen>
    </iframe>
    <div style="padding: 10px 14px; display: flex; justify-content: space-between; align-items: center; background: #212121; border-top: 1px solid #333;">
        <span style="font-size: 13px; color: #aaa;">Reference Track Example</span>
        <a href="{watch_url}" target="_blank" rel="noopener noreferrer" 
           style="background: #cc0000; color: white; padding: 6px 14px; border-radius: 4px; text-decoration: none; font-size: 13px; font-weight: bold; display: inline-flex; align-items: center; gap: 6px;">
            <span>▶</span> Watch on YouTube
        </a>
    </div>
</div>
    """
    return html