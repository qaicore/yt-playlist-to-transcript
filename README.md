# YouTube Playlist to Transcript Converter

A tool to convert any public YouTube Playlist into a markdown file of all the transcripts, with each one being separated by it's chapters within the video. 

This was built for LLMs to ingest multiple videos for a concrete breakdown of information, whilst maintaining specificty within each one. 

# Installation

```bash
git clone https://github.com/qaicore/yt-playlist-to-transcript.git
pip install yt-dlp
```

# Usage

Run this in the terminal:
```bash
python yt_playlist_transcripts.py "<playlist-url>" --out transcripts
```

Make sure your playlist is set to public on YouTube.