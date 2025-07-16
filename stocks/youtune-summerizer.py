from calendar import c
import os
from regex import F
import requests
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import argparse
import sys
import feedparser
import time

def get_youtube_transcript(video_url):
    """
    Downloads a YouTube video's transcript and returns it as a formatted string.
    """
    try:
        # Extract video ID from URL
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split('&')[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split('?')[0]
        else:
            print("Error: Invalid YouTube URL format.")
            return None

        # Attempt to get the transcript in Hebrew or English
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['he','iw', 'en'])
        
        formatted_transcript = ""
        for item in transcript_list:
            minutes = int(item['start']) // 60
            seconds = int(item['start']) % 60
            timestamp = f"{minutes:02d}:{seconds:02d}"
            formatted_transcript += f"[{timestamp}] {item['text']}\n"
            
        return formatted_transcript
    except Exception as e:
        print(f"❌ Error getting transcript: {e}")
        return None

def summarize_with_gemini(transcript):
    """
    Sends the transcript to Gemini and asks for a summary with timestamps.
    """
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
        return response.text.strip()
    except Exception as e:
        print(f"❌ Error generating summary with Gemini: {e}")
        return "Failed to generate summary."

def send_to_telegram(message, bot_token, chat_id):
    """
    Sends a message to Telegram using the provided bot token and chat ID.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML'
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("✅ Message sent to Telegram successfully!")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send message to Telegram: {e}")

def handle_new_video(video_url, telegram_token, telegram_chat_id):
    """
    Processes a single video: fetch transcript, summarize, and send to Telegram.
    Only returns True if successful.
    """
    try:
        print(f"Fetching transcript for: {video_url}...")
        transcript = get_youtube_transcript(video_url)
        
        if transcript:
            print("🧠 Transcript fetched. Summarizing with Gemini...")
            summary = summarize_with_gemini(transcript)
            print("✈️ Summary generated. Sending to Telegram...")
            final_message = f"📄 *סיכום עבור הסרטון:*\n`{video_url}`\n\n{summary}"
            send_to_telegram(final_message, telegram_token, telegram_chat_id)
            return True  # success
        else:
            print("Transcript not yet available. Will retry later.")
            return False  # failure
    except Exception as e:
        print(f"❌ Error handling new video: {e}")
        error_message = f"נכשלתי בעיבוד הסרטון:\n{video_url}\nשגיאה: {str(e)}"
        send_to_telegram(error_message, telegram_token, telegram_chat_id)
        return False

def watch_youtube_channel(channel_id, telegram_token, telegram_chat_id, check_interval=10):
    """
    Watch a YouTube channel for new videos and call handle_new_video().
    """
    rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    seen_videos = set()

    # # initialize with current videos
    # feed = feedparser.parse(rss_url)
    # for entry in feed.entries:
    #     seen_videos.add(entry.yt_videoid)

    print(f"👀 Watching YouTube channel: {channel_id}")

    while True:
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                video_id = entry.yt_videoid
                if video_id not in seen_videos:
                    video_url = f"https://youtu.be/{video_id}"
                    print(f"📹 New video detected: {video_url}")
                    success = handle_new_video(video_url, telegram_token, telegram_chat_id)
                    if success:
                        seen_videos.add(video_id)
        except Exception as e:
            print(f"❌ Error: {e}")
        time.sleep(check_interval)

def main():
    try:
        parser = argparse.ArgumentParser(
            description="Summarize a YouTube video and send the summary to Telegram.",
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        parser.add_argument("--video-url", help="The full URL of the YouTube video to summarize.", required=False)
        parser.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY"),
                            help="Gemini API Key. (or set GEMINI_API_KEY env var)")
        parser.add_argument("--telegram-token", default=os.getenv("TELEGRAM_BOT_TOKEN"),
                            help="Telegram Bot Token. (or set TELEGRAM_BOT_TOKEN env var)")
        parser.add_argument("--telegram-chat-id", default=os.getenv("TELEGRAM_CHAT_ID"),
                            help="Telegram Chat ID. (or set TELEGRAM_CHAT_ID env var)")
        parser.add_argument("--channal_id", default=os.getenv("YOUTUBE_CHANNEL_ID"),
                            help="YouTube Channel ID to watch for new videos.", required=False)
        args = parser.parse_args()

        if not all([args.gemini_key, args.telegram_token, args.telegram_chat_id]):
            print("🔴 Error: Missing credentials.")
            sys.exit(1)

        try:
            genai.configure(api_key=args.gemini_key)
        except Exception as e:
            print(f"🔴 Error configuring Gemini API: {e}")
            sys.exit(1)

        if not args.video_url:
            print("🔄 Watching YouTube channel for new videos...")
            watch_youtube_channel(args.channal_id, args.telegram_token, args.telegram_chat_id, check_interval=0.1)
        else:
            print(f"🔄 Processing video URL: {args.video_url}")
            handle_new_video(args.video_url, args.telegram_token, args.telegram_chat_id)
    except Exception as e:
        print(f"🔴 Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
