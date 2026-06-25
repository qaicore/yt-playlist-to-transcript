#!/usr/bin/env python3
"""
yt_playlist_transcripts.py

Scrape transcripts for every video in a YouTube playlist into a single combined
Markdown file, with each video clearly delimited and split under its chapter
("section") headings when it has them. Built for feeding a whole playlist into
an LLM for analysis.

Single external dependency: yt-dlp.

    pip install yt-dlp
    python yt_playlist_transcripts.py "<playlist-url>" --out transcripts

Design notes:
- yt-dlp enumerates the playlist, exposes per-video `chapters`, and hands back
  caption-track URLs. No second transcript library.
- Captions are pulled in YouTube's json3 format (one clean event per utterance
  with a precise start time). VTT is parsed only as a fallback.
- "Sections" == YouTube chapters: a separate timestamped list. Each transcript
  line is bucketed into the latest chapter that has already started.
- Fetching and formatting are separated. Each video is cached as JSON during
  the scrape (crash-safe + resumable), then the combined file is assembled from
  the cache. Re-run with a different --max-words to re-split without re-fetching.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import yt_dlp

def parse_json3(data: dict) -> list[tuple[float, str]]:
    """Turn a YouTube json3 caption payload into [(start_seconds, text), ...]."""
    lines: list[tuple[float, str]] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue  # spacing/formatting event, no words
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text:
            continue
        lines.append((ev.get("tStartMs", 0) / 1000.0, text))
    return lines

_VTT_TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")

def parse_vtt(text: str) -> list[tuple[float, str]]:
    """Fallback parser for WebVTT. Dedupes consecutive duplicate lines (the
    rolling-window artifact in YouTube auto-captions)."""
    lines: list[tuple[float, str]] = []
    cur_start: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if "-->" in line:
            m = _VTT_TS.search(line)
            if m:
                h, mm, s, ms = (int(x) for x in m.groups())
                cur_start = h * 3600 + mm * 60 + s + ms / 1000.0
            continue
        if not line or line == "WEBVTT" or line.isdigit():
            continue
        clean = re.sub(r"<[^>]+>", "", line).strip()  # strip inline timing tags
        if not clean or cur_start is None:
            continue
        if lines and lines[-1][1] == clean:
            continue
        lines.append((cur_start, clean))
    return lines

def chapter_index(start: float, chapters: list[dict]) -> int:
    """Index of the latest chapter that has started at time `start`."""
    idx = 0
    for i, ch in enumerate(chapters):
        if start >= ch.get("start_time", 0):
            idx = i
        else:
            break
    return idx

def bucket_by_chapters(
    transcript: list[tuple[float, str]],
    chapters: list[dict] | None,
) -> list[tuple[str | None, list[str]]]:
    """Group transcript lines under chapters. Returns [(title_or_None, [text])]."""
    if not chapters:
        return [(None, [t for _, t in transcript])]
    buckets: list[tuple[str | None, list[str]]] = [
        (ch.get("title") or f"Chapter {i + 1}", []) for i, ch in enumerate(chapters)
    ]
    for start, text in transcript:
        buckets[chapter_index(start, chapters)][1].append(text)
    return buckets

def sanitize_filename(name: str, max_len: int = 80) -> str:
    name = re.sub(r"[^\w\s.-]", "", name).strip()
    name = re.sub(r"\s+", "_", name)
    return name[:max_len] or "untitled"


def render_video(rec: dict) -> str:
    """Render one cached video record to Markdown."""
    out = [f"# [{rec['index']:02d}] {rec['title']}", "", f"<{rec['url']}>", ""]
    for heading, body in rec["sections"]:
        if not body:
            continue
        if heading is not None:
            out += [f"## {heading}", ""]
        out += [" ".join(body), ""]
    return "\n".join(out).rstrip()

def word_count(rec: dict) -> int:
    return sum(len(" ".join(body).split()) for _, body in rec["sections"])

def get_playlist(url: str) -> tuple[str, list[dict]]:
    """Fast flat listing. Returns (playlist_title, [{id, title}, ...])."""
    opts = {"quiet": True, "extract_flat": "in_playlist", "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries") or []
    if not entries and info.get("id"):  # a single video URL
        entries = [info]
    title = info.get("title") or "playlist"
    return title, [e for e in entries if e and e.get("id")]

def get_video_info(video_id: str) -> dict:
    opts = {"quiet": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)

def pick_caption_track(info: dict, lang: str, allow_auto: bool) -> tuple[str, str] | None:
    """(url, format) for the best caption track. Prefers manual subs and json3."""
    sources = [info.get("subtitles") or {}]
    if allow_auto:
        sources.append(info.get("automatic_captions") or {})
    for store in sources:
        keys = [k for k in store if k == lang] or [k for k in store if k.startswith(lang)]
        for key in keys:
            for fmt in ("json3", "srv3", "vtt"):
                for t in store[key]:
                    if t.get("ext") == fmt and t.get("url"):
                        return t["url"], fmt
    return None

def fetch_transcript(url: str, fmt: str) -> list[tuple[float, str]]:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read().decode("utf-8", errors="replace")
    if fmt in ("json3", "srv3") and raw.lstrip().startswith("{"):
        return parse_json3(json.loads(raw))
    return parse_vtt(raw)

def scrape(entries: list[dict], cache: Path, lang: str, allow_auto: bool,
           sleep: float, overwrite: bool) -> list[str]:
    """Fetch every video into cache/NNN.json. Returns titles that were skipped."""
    skipped: list[str] = []
    n = len(entries)
    for i, entry in enumerate(entries, 1):
        vid = entry["id"]
        title = entry.get("title") or vid
        cf = cache / f"{i:03d}.json"
        if cf.exists() and not overwrite:
            print(f"[{i}/{n}] cached: {title}")
            continue
        print(f"[{i}/{n}] {title}")
        try:
            info = get_video_info(vid)
            track = pick_caption_track(info, lang, allow_auto)
            if not track:
                print(f"    no '{lang}' captions — skipping")
                skipped.append(title)
                continue
            transcript = fetch_transcript(*track)
            if not transcript:
                print("    empty transcript — skipping")
                skipped.append(title)
                continue
            sections = bucket_by_chapters(transcript, info.get("chapters"))
            rec = {"index": i, "title": title,
                   "url": f"https://youtu.be/{vid}", "sections": sections}
            cf.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            n_sec = sum(1 for h, _ in sections if h is not None)
            print(f"    saved  ({len(transcript)} lines, "
                  f"{n_sec} sections)" if n_sec else f"    saved  ({len(transcript)} lines, no sections)")
        except Exception as e:  # one bad video shouldn't kill the run
            print(f"    error: {e}")
            skipped.append(title)
        if sleep:
            time.sleep(sleep)
    return skipped


def assemble(cache: Path, out_dir: Path, stem: str, max_words: int) -> None:
    """Build the combined file(s) from cached records."""
    records = [json.loads(p.read_text(encoding="utf-8"))
               for p in sorted(cache.glob("*.json"))]
    if not records:
        print("Nothing to assemble — no transcripts were captured.")
        return

    total = sum(word_count(r) for r in records)
    print(f"\n{len(records)} videos, {total:,} words "
          f"(~{total * 1.3:,.0f} tokens).")

    # Greedily pack whole videos into parts under the word budget.
    if max_words and total > max_words:
        parts: list[list[dict]] = [[]]
        running = 0
        for r in records:
            w = word_count(r)
            if running and running + w > max_words:
                parts.append([])
                running = 0
            parts[-1].append(r)
            running += w
        for pi, group in enumerate(parts, 1):
            body = "\n\n---\n\n".join(render_video(r) for r in group)
            gw = sum(word_count(r) for r in group)
            f = out_dir / f"{stem}_part{pi}.md"
            f.write_text(f"# {stem} — part {pi}/{len(parts)}\n\n{body}\n", encoding="utf-8")
            print(f"  {f.name}: {len(group)} videos, {gw:,} words")
    else:
        body = "\n\n---\n\n".join(render_video(r) for r in records)
        f = out_dir / f"{stem}.md"
        header = f"# {stem}\n\n{len(records)} videos, {total:,} words\n\n---\n\n"
        f.write_text(header + body + "\n", encoding="utf-8")
        print(f"  {f.name}")
        if total > 150_000:
            print("  (large — pass --max-words 120000 to split into context-sized parts)")


def run(url: str, out_dir: Path, lang: str, allow_auto: bool,
        sleep: float, overwrite: bool, max_words: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / ".cache"
    cache.mkdir(exist_ok=True)

    print(f"Resolving playlist: {url}")
    pl_title, entries = get_playlist(url)
    print(f"'{pl_title}' — {len(entries)} video(s).\n")

    skipped = scrape(entries, cache, lang, allow_auto, sleep, overwrite)
    assemble(cache, out_dir, sanitize_filename(pl_title), max_words)

    if skipped:
        print(f"\nSkipped {len(skipped)} (no usable captions): "
              + ", ".join(skipped[:5]) + ("..." if len(skipped) > 5 else ""))
    print(f"\nDone. Output in {out_dir}/  (per-video JSON cached in {cache}/)")

def main() -> None:
    p = argparse.ArgumentParser(description="Scrape a YouTube playlist into one combined, section-split transcript file.")
    p.add_argument("url", help="Playlist URL (or a single video URL)")
    p.add_argument("--out", default="transcripts", type=Path, help="Output directory (default: transcripts)")
    p.add_argument("--lang", default="en", help="Caption language prefix (default: en)")
    p.add_argument("--no-auto", action="store_true", help="Require manual subs; skip auto-generated captions")
    p.add_argument("--max-words", default=0, type=int, help="Split into parts of at most N words each (0 = single file)")
    p.add_argument("--sleep", default=1.0, type=float, help="Seconds between videos (default: 1.0)")
    p.add_argument("--overwrite", action="store_true", help="Re-fetch videos already in the cache")
    args = p.parse_args()
    try:
        run(args.url, args.out, args.lang, not args.no_auto, args.sleep, args.overwrite, args.max_words)
    except KeyboardInterrupt:
        print("\nInterrupted (cache preserved — re-run to resume).", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
