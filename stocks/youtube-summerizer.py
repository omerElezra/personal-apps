import os
import sys
import time
import argparse
import requests
import feedparser
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import logging
from logging.handlers import RotatingFileHandler


# Setup rotating log file
log_handler = RotatingFileHandler(
    filename="youtube_summary.log",
    maxBytes=5_000_000,  # 5 MB
    backupCount=3        # Keep last 3 logs
)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)

# Setup console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Root logger setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.addHandler(console_handler)


def get_youtube_transcript(video_url):
    try:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split('&')[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split('?')[0]
        else:
            logger.error("Invalid YouTube URL format.")
            return None

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['he', 'iw', 'en'])

        formatted_transcript = ""
        for item in transcript_list:
            minutes = int(item['start']) // 60
            seconds = int(item['start']) % 60
            timestamp = f"{minutes:02d}:{seconds:02d}"
            formatted_transcript += f"[{timestamp}] {item['text']}\n"

        logger.info(f"Transcript fetched for video {video_id}.")
        return formatted_transcript
    except Exception as e:
        logger.error(f"Error getting transcript: {e}")
        return None

def summarize_with_gemini(transcript):
    if not transcript:
        return "Could not generate summary because the transcript is empty."

    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    Please summarize the following YouTube video transcript in Hebrew.

    The summary should:
    - Be concise and clear, capturing the main points and key topics discussed in the video.
    - Be written as bullet points, each with an appropriate emoji or symbol to make it visually engaging.
    - Focus on key insights, stocks mentioned, trading tips, tickers, and strategies discussed in the video.
    - Be engaging and easy to read, suitable for a social media post.

    Additionally:
    - At the end of the summary, include actionable recommendations or practical insights based on the content of the video.
    - Add a call-to-action encouraging viewers to watch the full video for more details.
    
    Format:
    - Make sure the output is formatted as plain text and optimized for posting in a **Telegram message** (easy to read on mobile, using emojis and line breaks for readability).
    - Add the title of the video at the top, formatted as a header and the date of the video at the bottom.
    Transcript:
    {transcript}
    """
    
    try:
        response = model.generate_content(prompt)
        logger.info("Summary generated with Gemini.")
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating summary with Gemini: {e}")
        return "Failed to generate summary."

def send_to_telegram(message, bot_token, chat_id):
    chat_ids = [id.strip() for id in chat_id.split(',')]
    success_count = 0

    for single_chat_id in chat_ids:
        if not single_chat_id:
            continue

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {'chat_id': single_chat_id, 'text': message, 'parse_mode': 'HTML'}

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            success_count += 1
            logger.info(f"Message sent to Telegram chat {single_chat_id}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Telegram chat {single_chat_id}: {e}")
    
    if success_count > 0:
        logger.info(f"Message sent to {success_count}/{len(chat_ids)} channels successfully.")
    else:
        logger.error("Failed to send message to any Telegram channel.")

def handle_new_video(video_url, telegram_token, telegram_chat_id):
    try:
        logger.info(f"Processing video: {video_url}")
        transcript = get_youtube_transcript(video_url)

        if transcript:
            summary = summarize_with_gemini(transcript)
            final_message = f"📄 *Summary for video:*\n`{video_url}`\n\n{summary}"
            send_to_telegram(final_message, telegram_token, telegram_chat_id)
            return True
        else:
            logger.warning("Transcript not available yet. Will retry later.")
            return False
    except Exception as e:
        logger.error(f"Error handling video: {e}")
        # Only send errors to the first chat ID to avoid spamming all channels with errors
        first_chat_id = telegram_chat_id.split(',')[0].strip()
        error_message = f"⚠️ *Error processing video:*\n`{video_url}`\n\n{str(e)}"
        send_to_telegram(error_message, telegram_token, first_chat_id)
        return False

def watch_youtube_channel(channel_id, telegram_token, telegram_chat_id, check_interval=10):
    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    seen_videos = set()

    # Inintialize seen_videos with existing videos from the RSS feed for the first run
    feed = feedparser.parse(rss_url)
    for entry in feed.entries:
        seen_videos.add(entry.yt_videoid)

    logger.info(f"Watching YouTube channel {channel_id}.")

    while True:
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                video_id = entry.yt_videoid
                if video_id not in seen_videos:
                    video_url = f"https://youtu.be/{video_id}"
                    logger.info(f"New video detected: {video_url}")
                    success = handle_new_video(video_url, telegram_token, telegram_chat_id)
                    if success:
                        seen_videos.add(video_id)
        except Exception as e:
            logger.error(f"Error while watching channel: {e}")
        time.sleep(check_interval)

def main():
    parser = argparse.ArgumentParser(description="Summarize a YouTube video and send the summary to Telegram.")
    parser.add_argument("--video-url", help="YouTube video URL")
    parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY"), help="Gemini API Key")
    parser.add_argument("--telegram-token", default=os.getenv("TELEGRAM_BOT_TOKEN"), help="Telegram Bot Token")
    parser.add_argument("--telegram-chat-id", default=os.getenv("TELEGRAM_CHAT_ID"), help="Telegram Chat ID(s), comma-separated if multiple")
    parser.add_argument("--channel-id", default=os.getenv("YOUTUBE_CHANNEL_ID"), help="YouTube Channel ID")
    args = parser.parse_args()

    if not all([args.gemini_key, args.telegram_token, args.telegram_chat_id]):
        logger.error("Missing API credentials.")
        sys.exit(1)

    genai.configure(api_key=args.gemini_key)

    if args.video_url:
        handle_new_video(args.video_url, args.telegram_token, args.telegram_chat_id)
    else:
        watch_youtube_channel(args.channel_id, args.telegram_token, args.telegram_chat_id)


if __name__ == "__main__":
    main()