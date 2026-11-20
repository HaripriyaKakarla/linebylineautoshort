import asyncio
import json
import os
import time
import urllib.parse
from pathlib import Path

import edge_tts
import requests
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip, concatenate_videoclips
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "history.json"
CLIENT_SECRETS_FILE = BASE_DIR / "client_secrets.json"
TOKEN_FILE = BASE_DIR / "token.json"
OUTPUT_VIDEO_FILE = BASE_DIR / "short_output.mp4"

YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

NICHES = [
    "Personal Finance & Money Math (neutral math formulas, inflation, compound growth—NO financial advice)",
    "Everyday Physics & Mechanics (how objects move, balance, heat, and forces work)",
    "Psychology & Decision Biases (cognitive shortcuts, memory quirks, habit loops)",
    "Computer Science & Logic (algorithms, binary, sorting, networks explained simply)",
]

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def load_history() -> dict:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            print("-  History restored")
            return {
                "run_count": data.get("run_count", 0),
                "completed_topics": data.get("completed_topics", []),
            }
    return {"run_count": 0, "completed_topics": []}


def save_history(history: dict) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def generate_unique_script(niche: str, past_topics: list[str]) -> dict:
    """Generates a 4-scene educational script via Gemini with automatic retries and fallback."""
    past_list_str = ", ".join(past_topics[-100:]) if past_topics else "None yet"

    prompt = f"""
    Create an original, fact-checked 40-second educational YouTube Short script in this category:
    CATEGORY: {niche}

    CRITICAL RULE: Do NOT repeat or overlap with any of these previously covered topics:
    [{past_list_str}]

    Return ONLY valid JSON (no extra commentary) with this exact schema:
    {{
      "topic_id": "Short 3-5 word topic name for deduplication log",
      "title": "Catchy YouTube Short title under 60 chars #Shorts",
      "description": "2-sentence educational summary + relevant hashtags #Shorts",
      "tags": ["education", "explained", "learning", "Shorts"],
      "scenes": [
        {{
          "narration": "Hook sentence (15-22 words) that grabs curiosity immediately.",
          "visual_prompt": "Describe one simple stick-figure scene illustrating this exact sentence."
        }},
        {{
          "narration": "Core explanation sentence 1 (20-28 words).",
          "visual_prompt": "Describe one simple stick-figure scene illustrating this exact sentence."
        }},
        {{
          "narration": "Concrete example or mechanism sentence 2 (20-28 words).",
          "visual_prompt": "Describe one simple stick-figure scene illustrating this exact sentence."
        }},
        {{
          "narration": "Crisp wrap-up takeaway sentence (15-22 words).",
          "visual_prompt": "Describe one simple stick-figure scene illustrating the takeaway."
        }}
      ]
    }}
    """

    models = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
    max_retries_per_model = 4
    last_error = None

    for model_name in models:
        for attempt in range(1, max_retries_per_model + 1):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    ),
                )
                return json.loads(response.text)
            except Exception as err:
                last_error = err
                err_str = str(err)
                is_transient = (
                    "503" in err_str
                    or "429" in err_str
                    or "UNAVAILABLE" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                )
                if is_transient and attempt < max_retries_per_model:
                    sleep_seconds = 5 * attempt
                    print(
                        f"[{model_name}] Transient error (attempt {attempt}/{max_retries_per_model}). "
                        f"Retrying in {sleep_seconds}s..."
                    )
                    time.sleep(sleep_seconds)
                else:
                    print(f"[{model_name}] Moving on due to error: {err}")
                    break

    raise RuntimeError(f"All Gemini attempts failed. Last error: {last_error}")


def _draw_fallback_stick_scene(
    scene_index: int,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """Draws a clean black-ink stick figure on an off-white 9:16 canvas as a fallback."""
    img = Image.new("RGB", (width, height), color=(248, 245, 238))
    draw = ImageDraw.Draw(img)
    ink = (25, 25, 25)
    accent = (40, 90, 180)
    lw = 10

    ground_y = int(height * 0.68)
    draw.line([(120, ground_y), (width - 120, ground_y)], fill=ink, width=lw)

    cx = int(width * 0.35)
    head_r = 65
    head_cy = ground_y - 340
    torso_top = head_cy + head_r
    torso_bot = ground_y - 130

    draw.ellipse(
        [(cx - head_r, head_cy - head_r), (cx + head_r, head_cy + head_r)],
        outline=ink,
        width=lw,
    )
    draw.line([(cx, torso_top), (cx, torso_bot)], fill=ink, width=lw)

    arm_y = torso_top + 40
    if scene_index % 2 == 0:
        draw.line([(cx - 130, arm_y - 35), (cx, arm_y), (cx + 135, arm_y - 55)], fill=ink, width=lw)
    else:
        draw.line([(cx - 120, arm_y + 30), (cx, arm_y), (cx + 140, arm_y - 70)], fill=ink, width=lw)

    draw.line([(cx, torso_bot), (cx - 90, ground_y)], fill=ink, width=lw)
    draw.line([(cx, torso_bot), (cx + 95, ground_y)], fill=ink, width=lw)

    bx = int(width * 0.62)
    for i in range(3):
        bar_h = 70 + i * 55
        top_y = ground_y - bar_h
        draw.rectangle(
            [(bx + i * 95, top_y), (bx + i * 95 + 65, ground_y)],
            outline=accent,
            width=7,
        )

    img.save(output_path)
    return output_path


def generate_stick_drawing(
    visual_prompt: str,
    scene_index: int,
    output_path: Path,
    seed: int = 1000,
) -> Path:
    """Generates a clean full-bleed stick-figure image via Pollinations with Pillow fallback."""
    style_suffix = (
        ", minimalist black ink stick figure drawing filling the entire 9:16 frame, "
        "clean off-white paper background edge to edge, no wall, no borders, no pins, "
        "expressive hand-drawn whiteboard doodle style, simple geometric props, "
        "high contrast, strictly no text, no words, no letters, no numbers, no watermarks"
    )
    full_prompt = f"{visual_prompt}{style_suffix}"
    encoded_prompt = urllib.parse.quote(full_prompt, safe="")

    api_key = os.environ.get("POLLINATIONS_API_KEY", "").strip()
    headers = {}
    params = {
        "model": "flux",
        "width": "1080",
        "height": "1920",
        "seed": str(seed + scene_index * 31),
        "nologo": "true",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        params["key"] = api_key

    url = f"https://gen.pollinations.ai/image/{encoded_prompt}"

    try:
        response = requests.get(url, headers=headers, params=params, timeout=75)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type and not response.content:
            raise ValueError(f"Unexpected non-image response: {content_type}")

        with open(output_path, "wb") as f:
            f.write(response.content)

        with Image.open(output_path) as check_img:
            check_img.verify()

        return output_path
    except Exception as exc:
        print(
            f"[Scene {scene_index}] Pollinations image error ({exc}). "
            f"Using local Pillow stick-figure fallback."
        )
        return _draw_fallback_stick_scene(scene_index=scene_index, output_path=output_path)


async def _synthesize_voice(text: str, mp3_path: Path, voice: str = "en-US-ChristopherNeural") -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(mp3_path))


def build_video(script: dict) -> Path:
    """Generates scene images + narration audio and assembles a 9:16 YouTube Short."""
    scene_clips = []
    audio_clips = []
    temp_files = []

    try:
        for idx, scene in enumerate(script["scenes"]):
            img_path = BASE_DIR / f"scene_{idx}.png"
            aud_path = BASE_DIR / f"scene_{idx}.mp3"
            temp_files.extend([img_path, aud_path])

            generate_stick_drawing(
                visual_prompt=scene["visual_prompt"],
                scene_index=idx,
                output_path=img_path,
                seed=5000 + idx * 137,
            )
            asyncio.run(_synthesize_voice(scene["narration"], aud_path))

            audio_clip = AudioFileClip(str(aud_path))
            audio_clips.append(audio_clip)

            duration = max(audio_clip.duration + 0.25, 2.0)

            # Subtle slow zoom from 1.0x to 1.05x locked inside a 1080x1920 canvas
            base_clip = ImageClip(str(img_path)).resized((1080, 1920)).with_duration(duration)
            zoomed_clip = (
                base_clip.resized(lambda t, d=duration: 1.0 + 0.05 * (t / max(d, 0.1)))
                .with_position("center")
                .with_duration(duration)
            )
            framed_clip = (
                CompositeVideoClip([zoomed_clip], size=(1080, 1920))
                .with_duration(duration)
                .with_audio(audio_clip)
            )
            scene_clips.append(framed_clip)

        final_video = concatenate_videoclips(scene_clips, method="compose")
        final_video.write_videofile(
            str(OUTPUT_VIDEO_FILE),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger=None,
        )

        final_video.close()
        for clip in scene_clips:
            clip.close()
        for aclip in audio_clips:
            aclip.close()

        return OUTPUT_VIDEO_FILE
    finally:
        for f in temp_files:
            if f.exists():
                try:
                    f.unlink()
                except OSError:
                    pass


def get_youtube_service():
    """Authenticates via token.json or runs one-time browser OAuth using client_secrets.json."""
    creds = None
    token_env_json = os.environ.get("YOUTUBE_TOKEN_JSON", "").strip()

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), YOUTUBE_SCOPES)
    elif token_env_json:
        creds = Credentials.from_authorized_user_info(json.loads(token_env_json), YOUTUBE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            from google.auth.transport.requests import Request

            creds.refresh(Request())
        else:
            if not CLIENT_SECRETS_FILE.exists():
                raise FileNotFoundError(
                    f"Missing {CLIENT_SECRETS_FILE}. Place client_secrets.json in {BASE_DIR} to generate token.json."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_FILE), YOUTUBE_SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w", encoding="utf-8") as token_out:
            token_out.write(creds.to_json())
        print(f"Saved OAuth token to {TOKEN_FILE}")

    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(video_path: Path, script: dict) -> str:
    youtube = get_youtube_service()
    body = {
        "snippet": {
            "title": script["title"][:100],
            "description": script["description"],
            "tags": script.get("tags", ["Shorts", "education"]),
            "categoryId": "27",  # Education
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        _, response = request.next_chunk()

    video_id = response.get("id", "")
    print(f"Uploaded video ID: {video_id}")
    return video_id


def main() -> None:
    history = load_history()
    niche_today = NICHES[history["run_count"] % len(NICHES)]
    print(f"Today's Niche Pillar: {niche_today}")

    script = generate_unique_script(niche_today, history["completed_topics"])
    print(f"Generated Unique Topic: {script['topic_id']} -> {script['title']}")

    video_path = build_video(script)
    upload_to_youtube(video_path, script)

    history["run_count"] += 1
    history["completed_topics"].append(script["topic_id"])
    save_history(history)

    if video_path.exists():
        try:
            video_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()