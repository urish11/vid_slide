import streamlit as st
import pandas as pd
import fal_client
import anthropic
from openai import OpenAI
import tempfile
#from pydub import AudioSegment # Retained but not actively used due to commented out boost
import os
import logging
import time
import requests # For downloading the image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import moviepy.editor as mp
import json
import math
from collections.abc import Callable
import boto3
from io import BytesIO # Not directly used for video file upload, but good S3 utility
import random
import os
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS
st.set_page_config(layout="wide", page_title="Vid Slide Gen",page_icon="🎦")

# --- Configuration for Logging ---
# Streamlit typically handles its own logging display.
# Console logging is still useful for development/debugging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initial API Key Placeholders (User will override these in Streamlit UI) ---
# It's better practice to not have default keys in code.
# These are from your script, will be replaced by st.session_state values.
# 🧠 API Keys
openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
# ☁️ S3 Configuration
s3_bucket_name = st.secrets.get("S3_BUCKET_NAME", "")
aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID", "")
aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY", "")
s3_region = st.secrets.get("S3_REGION_NAME", "us-east-1")  # Default fallback
os.environ["FAL_KEY"] =  st.secrets.get("FAL_KEY")
# --- Helper for Fal Client ---
def on_queue_update(update):
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            logging.info(f"[FalClient Log] {log['message']}")
            # st.sidebar.text(f"[Fal Log] {log['message']}") # Optional: log to sidebar

# --- Image Downloader ---
def download_image(image_url: str, save_path: str) -> bool:
    """Downloads an image from a URL and saves it locally."""
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Image downloaded successfully to {save_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download image from {image_url}: {e}")
        st.warning(f"Failed to download image: {image_url}")
        return False

# --- 1. Background Image Generation (Fal) ---
def zoom_effect(
    clip: mp.VideoClip,
    ratio: float = 0.04,
) -> mp.VideoClip:
    """
    Apply a zoom effect to a clip.
    """
    def _apply(
        get_frame: Callable[[float], np.ndarray],
        t: float,
    ) -> np.ndarray:
        img = Image.fromarray(get_frame(t))
        base_size = img.size
        new_size = (
            math.ceil(img.size[0] * (1 + (ratio * t))),
            math.ceil(img.size[1] * (1 + (ratio * t))),
        )
        new_size = (
            new_size[0] + (new_size[0] % 2),
            new_size[1] + (new_size[1] % 2),
        )
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        x = math.ceil((new_size[0] - base_size[0]) / 2)
        y = math.ceil((new_size[1] - base_size[1]) / 2)
        img = img.crop((x, y, new_size[0] - x, new_size[1] - y)).resize(
            base_size, Image.Resampling.LANCZOS
        )
        result = np.array(img)
        img.close()
        return result
    return clip.fl(_apply)

def generate_fal_image(full_prompt: str): # Changed 'topic' to 'full_prompt'
    logging.info(f"--- Requesting image from Fal with prompt: {full_prompt[:100]}... ---")
    st.write(f"Fal: Generating image for prompt: {full_prompt[:150]}...")
    try:
        result = fal_client.subscribe(
            "rundiffusion-fal/juggernaut-flux/lightning", # Using a potentially faster/cheaper model as an example
            # "rundiffusion-fal/juggernaut-flux/lightning", # Original model
            arguments={
                "prompt": full_prompt, # Use the full prompt directly
                "image_size": "portrait_16_9", # Or "square_hd" / "landscape_16_9"
                "num_inference_steps": 12, # Fast, adjust if quality needed
                "num_images": 1,
                "enable_safety_checker": True
            },
            with_logs=True, # Set to False to reduce console noise if preferred
            on_queue_update=on_queue_update
        )
        logging.info(f"Fal image generation result: {result}")
        if result and 'images' in result and len(result['images']) > 0:
            st.write("Fal: Image generated.")
            return result['images'][0]
        else:
            logging.error("No image data found in Fal result.")
            st.warning("Fal: No image data returned.")
            return None
    except Exception as e:
        logging.error(f"Error during Fal image generation: {e}")
        st.error(f"Fal Error: {e}")
        return None

# --- 2. Text Generation with Claude ---
def generate_text_with_claude(prompt: str, anthropic_api_key: str, model: str = "claude-sonnet-4-20250514", temperature: float = 0.88, max_retries: int = 3): # claude-3-opus-20240229, claude-3-sonnet-20240229, claude-3-haiku-20240307
    logging.info(f"--- Requesting text from Claude with prompt: '{prompt[:70]}...' ---")
    st.write(f"Claude: Generating text (model: {model})...")
    tries = 0
    while tries < max_retries:
        try:
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            message_payload = {
                "model": model,
                "max_tokens": 2048, # Increased for potentially longer prompts or JSON
                "temperature": temperature,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

            }
            response = client.messages.create(**message_payload)
            if len(response.content) > 0 and response.content[0].type == "text":
                generated_text = response.content[0].text
                logging.info(f"Claude generated text: {generated_text[:100]}...")
                st.write(f"Claude: Text generated: {generated_text}")
                return generated_text
            else:
                logging.error("Claude response content not found or not text.")
                st.warning("Claude: Response content issue.")
                generated_text = "" # Return empty string for this attempt
        except anthropic.APIConnectionError as e:
            logging.error(f"Claude APIConnectionError (attempt {tries + 1}/{max_retries}): {e}")
            st.warning(f"Claude connection error (attempt {tries+1}), retrying...")
        except anthropic.RateLimitError as e:
            logging.error(f"Claude RateLimitError (attempt {tries + 1}/{max_retries}): {e}")
            st.warning(f"Claude rate limit hit (attempt {tries+1}), retrying after delay...")
            time.sleep(15 if tries < 2 else 30) # Longer sleep for rate limits
        except anthropic.APIStatusError as e:
            logging.error(f"Claude APIStatusError status={e.status_code} (attempt {tries + 1}/{max_retries}): {e.message}")
            st.error(f"Claude API error {e.status_code} (attempt {tries+1}): {e.message}")
        except Exception as e:
            logging.error(f"Error during Claude text generation (attempt {tries + 1}/{max_retries}): {e}")
            st.error(f"Claude general error (attempt {tries+1}): {e}")
        
        tries += 1
        if tries < max_retries:
            time.sleep(5 * tries) # Exponential backoff
        else:
            logging.error("Max retries reached for Claude.")
            st.error("Claude: Max retries reached. Failed to generate text.")
            return None
    return None # Should be unreachable if loop logic is correct, but as a fallback


# --- 3. TTS Audio Generation (OpenAI) ---
def generate_audio_with_timestamps(text: str, openai_client: OpenAI, voice_id: str = "sage"):
    logging.info(f"--- Generating audio for text: '{text[:50]}...' with voice: {voice_id} ---")
    st.write(f"OpenAI TTS: Generating audio with voice '{voice_id}'...")
    temp_audio_path = None
    try:
        if not text or not text.strip():
            raise ValueError("Input text for TTS cannot be empty.")
        tts_model = "gpt-4o-mini-tts"
        instructions_per_voice = {
            'redneck': {'instructions': 'talk like an older american redneck heavy accent. deep voice, enthusiastic', 'voice': 'ash'},
            'announcer': {'instructions': 'Polished announcer voice, American accent', 'voice': 'ash'},
            'sage': {'instructions': 'high energy enthusiastic', 'voice': 'sage'},
            'announcer uk': {'instructions': 'Polished announcer voice, British accent', 'voice': 'ash'}
        }
        
        openai_voice_to_use = voice_id
        speech_params = {"input": text, "response_format": "mp3", "speed": 1.05}
        tts_model = "gpt-4o-mini-tts"
        if voice_id in instructions_per_voice:
            tts_model = "gpt-4o-mini-tts" # This model name is not standard for OpenAI TTS API
            # The user's original code had a commented out check for gpt-4o-mini-tts. Standard models are tts-1, tts-1-hd.
            # If using a specific model that supports instructions, this is where it would be set.
            # For now, assuming tts-1 or tts-1-hd.
            openai_voice_to_use = instructions_per_voice[voice_id]['voice']
            # Instructions are not a standard parameter for openai.audio.speech.create
            # If this was for a different API or a hypothetical feature, it would go here.
            # For standard OpenAI TTS, we control voice and speed. The 'prompt' for TTS is the text itself.
            # speech_params["instructions"] = instructions_per_voice[voice_id]['instructions']
            speech_params["instructions"] = instructions_per_voice[voice_id]['instructions']
            speech_params["speed"] = instructions_per_voice[voice_id].get('speed', 1.0)
        
        speech_params["model"] = tts_model
        speech_params["voice"] = openai_voice_to_use
        
        response = openai_client.audio.speech.create(**speech_params)
        st.text(response)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file_obj:
            temp_audio_path = temp_audio_file_obj.name
            temp_audio_file_obj.write(response.content)
        logging.info(f"Temporary audio file saved at: {temp_audio_path}")
        st.write("OpenAI TTS: Audio generated.")
        
        # Volume boost was commented out, kept as is.
        # try:
        #     boosted_audio = AudioSegment.from_file(temp_audio_path)
        #     boosted_audio = boosted_audio + 8 # Boost by 8dB
        #     boosted_audio.export(temp_audio_path, format="mp3")
        #     logging.info(f"Audio volume boosted for {temp_audio_path}")
        # except Exception as boost_err:
        #     logging.warning(f"Could not boost audio volume: {boost_err}. Using original audio.")

        # Timestamp generation was commented out, kept as is.
        # Word timings are not being returned or used by the rest of the script.
        return temp_audio_path
        
    except Exception as e:
        logging.error(f"OpenAI API Error in TTS: {e}")
        st.error(f"OpenAI TTS Error: {e}")
        if temp_audio_path and os.path.exists(temp_audio_path):
            try: os.remove(temp_audio_path)
            except Exception as rm_err: logging.warning(f"Could not remove temp audio file {temp_audio_path}: {rm_err}")
        return None


# --- 4. MoviePy Visual Generation Functions ---
def make_rounded_rect_png(size, radius, fill=(0, 0, 0, 255)):
    w, h = size
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rounded_rectangle([(0, 0), (w, h)], radius=radius, fill=fill)
    return np.array(img)
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_font_path(font_filename, base_path=None):
    """Gets the absolute path to a font file bundled with the app."""
    if base_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of the current script

    # Try in a 'fonts' subdirectory first
    fonts_dir_path = os.path.join(base_path, "fonts", font_filename)
    if os.path.exists(fonts_dir_path):
        return fonts_dir_path

    # Try in the root directory of the script
    root_path = os.path.join(base_path, font_filename)
    if os.path.exists(root_path):
        return root_path

    logging.warning(f"Font '{font_filename}' not found in 'fonts/' or root directory.")
    return font_filename # Fallback to name, hoping system might find it (unlikely for custom fonts on cloud)

def rounded_bg_text_pillow(text, font_filename="boogaloo.ttf", fontsize=100, text_color="white",
                           bg_color=(0, 0, 0, 255), radius=40, pad_x=40, pad_y=20, duration=4,
                           stroke_width=0, stroke_fill="black"): # Optional text stroke
    try:
        font_path = get_font_path(font_filename)
        font = ImageFont.truetype(font_path, fontsize)
    except IOError as e:
        logging.warning(f"Pillow: Font '{font_filename}' (path: {font_path}) failed to load: {e}. Trying Liberation Sans.")
        st.warning(f"Font '{font_filename}' not found or error. Trying 'LiberationSans-Regular.ttf'. Text: '{text}'")
        try:
            font_path = get_font_path("LiberationSans-Regular.ttf") # Ensure this is also in your repo
            font = ImageFont.truetype(font_path, fontsize)
        except IOError as e2:
            logging.error(f"Pillow: Fallback font 'LiberationSans-Regular.ttf' also failed: {e2}. Using default.")
            st.error(f"Critical font error for Text rendering (Pillow). Default font will be used. Text: '{text}'")
            font = ImageFont.load_default() # Very basic, likely not what you want visually

    # Calculate text size
    # Create a dummy draw object to calculate text bounding box
    dummy_img = Image.new("RGB", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)

    try:
        # For Pillow >= 9.2.0, textbbox is preferred for more accuracy
        bbox = dummy_draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        # Offset to draw from the actual top-left of the text pixels
        text_draw_x_offset = -bbox[0]
        text_draw_y_offset = -bbox[1]
    except AttributeError:
        # Fallback for older Pillow versions (textsize is less accurate)
        text_w, text_h = dummy_draw.textsize(text, font=font, stroke_width=stroke_width)
        text_draw_x_offset = 0
        text_draw_y_offset = 0


    bg_w = int(text_w + pad_x * 2)
    bg_h = int(text_h + pad_y * 2)

    # Create background image
    bg_pil_img = Image.new("RGBA", (bg_w, bg_h), (0, 0, 0, 0)) # Transparent background
    draw = ImageDraw.Draw(bg_pil_img)
    draw.rounded_rectangle([(0, 0), (bg_w, bg_h)], radius=radius, fill=bg_color)

    # Position to draw text on the background (considering padding and text's own bbox)
    text_final_x = pad_x + text_draw_x_offset
    text_final_y = pad_y + text_draw_y_offset

    # Draw text
    draw.text((text_final_x, text_final_y), text, font=font, fill=text_color,
              stroke_width=stroke_width, stroke_fill=stroke_fill)

    # Convert Pillow image to NumPy array for MoviePy
    frame_np = np.array(bg_pil_img)

    # Create MoviePy ImageClip
    clip = mp.ImageClip(frame_np, ismask=False).set_duration(duration)
    return clip


def elastic_out(t):
    c4 = (2 * np.pi) / 3
    if t == 0: return 0
    if t == 1: return 1
    return pow(2, -10 * t) * np.sin((t * 10 - 0.75) * c4) + 1

def ease_out_back(t, c1=1.70158, c3=None):
    if c3 is None: c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)

def create_facebook_ad_new(bg_img_path: str, headline_text1, headline_text2, headline_text3, duration: int = 7, resolution=(1080, 1920), learn_more = "Learn More Now"):
    logging.info(f"--- Creating Facebook Ad visuals with background: {bg_img_path} ---")
    st.write("MoviePy: Creating video visuals...")
    fps = 30
    frame_width, frame_height = resolution[0], resolution[1]
    final_clip = None
    background_clip_obj = None
    text_clip1_obj, text_clip2_obj, text_clip3_obj, button_clip_obj = None, None, None, None

    try:
        if bg_img_path and os.path.exists(bg_img_path):
            try:
                background_clip_obj = mp.ImageClip(bg_img_path)
                background_clip_obj = zoom_effect(background_clip_obj, 0.035)
            except Exception as e:
                logging.error(f"Error loading background image '{bg_img_path}': {e}")
                st.warning(f"MoviePy: Error loading background image '{os.path.basename(bg_img_path)}'. Using black fallback.")
                background_clip_obj = mp.ColorClip(size=resolution, color=(0,0,0), duration=duration)
        else:
            logging.warning(f"Background image path invalid or not provided ('{bg_img_path}'). Using black background.")
            st.warning("MoviePy: Background image not found. Using black fallback.")
            background_clip_obj = mp.ColorClip(size=resolution, color=(0,0,0), duration=duration)

        clip_aspect_ratio = background_clip_obj.w / background_clip_obj.h
        frame_aspect_ratio = frame_width / frame_height
        if clip_aspect_ratio > frame_aspect_ratio:
            scaled_background = background_clip_obj.resize(height=frame_height)
        else:
            scaled_background = background_clip_obj.resize(width=frame_width)
        
        background_final = scaled_background.crop(
            x_center=scaled_background.w / 2, y_center=scaled_background.h / 2,
            width=frame_width, height=frame_height
        ).set_duration(duration)

        text_color = 'yellow'
        button_text = learn_more # This could be made dynamic / language-specific

        text_clip1_obj = rounded_bg_text_pillow(headline_text1, font_filename="boogaloo.ttf", fontsize=90, text_color=text_color, bg_color=(0,0,0,220), radius=50, pad_x=50, pad_y=25, duration=duration)

        text_clip1_final_y = resolution[1] * 0.15
        text_clip2_obj = rounded_bg_text_pillow(headline_text2, font_filename="boogaloo.ttf", fontsize=90, text_color=text_color, bg_color=(0,0,0,220), radius=50, pad_x=50, pad_y=25, duration=duration)

        text_clip2_final_y = text_clip1_final_y + text_clip1_obj.h + 25
        text_clip3_obj = rounded_bg_text_pillow(headline_text3, font_filename="boogaloo.ttf", fontsize=90, text_color=text_color, bg_color=(0,0,0,220), radius=50, pad_x=50, pad_y=25, duration=duration)
        text_clip3_final_y = text_clip2_final_y + text_clip2_obj.h + 25

        button_bg_color = (0, 0, 200) # Darker blue
        # try:
        #     button_text_render = mp.TextClip(button_text, fontsize=65, color=text_color, font="Arial-Bold", method='label').set_duration(duration)
        # except Exception as e:
        #     st.warning("MoviePy: Font 'Arial-Bold' not found for button. Trying 'Liberation-Sans'.")
        #     try:
        #         button_text_render = mp.TextClip(button_text, fontsize=65, color=text_color, font="Liberation-Sans", method='label').set_duration(duration)
        #     except:
        #         button_text_render = mp.TextClip(button_text, fontsize=65, color=text_color, method='label').set_duration(duration)


        # button_width, button_height = button_text_render.w + 80, button_text_render.h + 40
        # button_bg = mp.ColorClip(size=(button_width, button_height), color=button_bg_color, ismask=False, duration=duration)
        # button_text_render = button_text_render.set_position(('center', 'center'))
        # button_clip_obj = mp.CompositeVideoClip([button_bg, button_text_render], size=(button_width, button_height)).set_duration(duration)
        # button_final_y = resolution[1] * 0.65 - button_height / 2 # Adjusted y-position


        button_fontsize = 65
        button_font_name = "boogaloo.ttf" # Example: ensure this font is in your repo
        try:
            button_actual_font_path = get_font_path(button_font_name)
            pil_button_font = ImageFont.truetype(button_actual_font_path, button_fontsize)
        except IOError:
            logging.warning(f"Button font {button_font_name} not found. Using LiberationSans-Regular.")
            st.warning(f"Button font '{button_font_name}' not found. Using fallback.")
            button_actual_font_path = get_font_path("LiberationSans-Regular.ttf") # Fallback
            try:
                pil_button_font = ImageFont.truetype(button_actual_font_path, button_fontsize)
            except IOError:
                logging.error("Fallback button font also not found. Using default PIL font.")
                pil_button_font = ImageFont.load_default()

        dummy_img_btn = Image.new("RGB", (1,1))
        dummy_draw_btn = ImageDraw.Draw(dummy_img_btn)
        try:
            btn_bbox = dummy_draw_btn.textbbox((0,0), button_text, font=pil_button_font)
            btn_text_w = btn_bbox[2] - btn_bbox[0]
            btn_text_h = btn_bbox[3] - btn_bbox[1]
            btn_draw_x_offset = -btn_bbox[0]
            btn_draw_y_offset = -btn_bbox[1]
        except AttributeError:
            btn_text_w, btn_text_h = dummy_draw_btn.textsize(button_text, font=pil_button_font)
            btn_draw_x_offset = 0
            btn_draw_y_offset = 0

        # Create transparent PIL image for the text itself
        button_text_pil_img = Image.new("RGBA", (int(btn_text_w), int(btn_text_h)), (0,0,0,0))
        draw_button_text = ImageDraw.Draw(button_text_pil_img)
        draw_button_text.text((btn_draw_x_offset, btn_draw_y_offset), button_text, font=pil_button_font, fill=text_color)

        button_text_render_np = np.array(button_text_pil_img)
        button_text_render = mp.ImageClip(button_text_render_np).set_duration(duration)
        # button_text_render now IS the text clip (transparent background)

        # The rest of your button composition logic:
        button_width, button_height = button_text_render.w + 80, button_text_render.h + 40
        button_bg = mp.ColorClip(size=(int(button_width), int(button_height)), color=button_bg_color, ismask=False, duration=duration)
        button_text_render = button_text_render.set_position(('center', 'center'))
        button_clip_obj = mp.CompositeVideoClip([button_bg, button_text_render], size=(int(button_width), int(button_height))).set_duration(duration)
        button_final_y = resolution[1] * 0.52 - button_height / 2 # Adjusted y-position || was 0.65


        time_multi = 1.5
        start_time = 0.6
        anim_dur_line1, anim_dur_line2, anim_dur_line3 = 0.7 * time_multi, 0.6 * time_multi, 0.6 * time_multi
        anim_dur_rv_pop = 0.4 * time_multi # This var seems unused later.
        pulse_start_delay = 0.3
        line2_start_time = start_time + anim_dur_line1 * 0.3
        line3_start_time = line2_start_time + anim_dur_line2 * 0.5
        text1_anim_end_time = start_time + anim_dur_line1
        text2_anim_end_time = line2_start_time + anim_dur_line2
        text3_anim_end_time = line3_start_time + anim_dur_line3
        max_text_settle_time = max(text1_anim_end_time, text2_anim_end_time, text3_anim_end_time)
        ELEGANT_FLOAT_START_DELAY, ELEGANT_FLOAT_AMPLITUDE, ELEGANT_FLOAT_PERIOD = 0.5, 7, 4.0
        actual_float_start_time = max_text_settle_time + ELEGANT_FLOAT_START_DELAY

        def animate_pos(t, clip_h, final_y, anim_start_time, anim_duration, anim_end_time, ease_func=ease_out_back, entry_from='top', x_align='center' , clip_width_val=None):
            y_start_top = -clip_h
            x_start_left = -resolution[0]
            x_start_right = resolution[0]
            
            actual_clip_width = clip_width_val if clip_width_val is not None else resolution[0] # Default to full if not provided
            x_final = (resolution[0] - actual_clip_width) / 2 if x_align == 'center' else 0 # Simplified for center
            if x_align != 'center' and entry_from == 'left': x_final = 0 # align to left if sliding from left
            if x_align != 'center' and entry_from == 'right': x_final = resolution[0] - actual_clip_width # align to right

            current_x, current_y = x_final, final_y

            if entry_from == 'top': current_x = 'center'
            
            if t < anim_start_time:
                if entry_from == 'top': return (current_x, y_start_top)
                if entry_from == 'left': return (x_start_left, final_y)
                if entry_from == 'right': return (x_start_right, final_y)
            
            if t < anim_end_time:
                anim_progress = np.clip((t - anim_start_time) / anim_duration, 0, 1)
                eased_progress = ease_func(anim_progress)

                if entry_from == 'top':
                    y_overshoot = final_y + 30
                    if anim_progress < 0.5:
                        current_y = y_start_top + (y_overshoot - y_start_top) * (anim_progress * 2)
                    else:
                        bounce_progress = (anim_progress - 0.5) * 2
                        current_y = y_overshoot - 30 * elastic_out(bounce_progress)
                elif entry_from == 'left':
                    current_x = x_start_left + (x_final - x_start_left) * eased_progress
                elif entry_from == 'right':
                    current_x = x_start_right + (x_final - x_start_right) * eased_progress
                return (current_x, current_y)

            if t < actual_float_start_time:
                if entry_from == 'top': return ('center', final_y)
                return (x_final, final_y)
            else:
                time_into_float = t - actual_float_start_time
                delta_y = ELEGANT_FLOAT_AMPLITUDE * np.sin((2 * np.pi / ELEGANT_FLOAT_PERIOD) * time_into_float)
                if entry_from == 'top': return ('center', final_y + delta_y)
                if entry_from == 'right':
                     delta_y = ELEGANT_FLOAT_AMPLITUDE * 0.8 * np.sin((2 * np.pi / ELEGANT_FLOAT_PERIOD) * time_into_float + np.pi/4)
                return (x_final, final_y + delta_y)

        text_clip1_obj = text_clip1_obj.set_position(lambda t: animate_pos(t, text_clip1_obj.h, text_clip1_final_y, start_time, anim_dur_line1, text1_anim_end_time, entry_from='top', clip_width_val=text_clip1_obj.w))
        text_clip2_obj = text_clip2_obj.set_position(lambda t: animate_pos(t, text_clip2_obj.h, text_clip2_final_y, line2_start_time, anim_dur_line2, text2_anim_end_time, entry_from='left', clip_width_val=text_clip2_obj.w))
        text_clip3_obj = text_clip3_obj.set_position(lambda t: animate_pos(t, text_clip3_obj.h, text_clip3_final_y, line3_start_time, anim_dur_line3, text3_anim_end_time, ease_func=lambda t_prog: ease_out_back(t_prog, c1=0.8), entry_from='right', clip_width_val=text_clip3_obj.w))
        
        button_anim_start_time = max_text_settle_time + 0.5 # Button appears after text settles
        original_button_size = button_clip_obj.size

        def animate_button_size(t):
            if t < button_anim_start_time:
                return (1, 1) # Effectively invisible by being tiny

            pop_duration = 0.3
            pop_end_time = button_anim_start_time + pop_duration
            
            if t < pop_end_time:
                progress = (t - button_anim_start_time) / pop_duration
                scale = progress 
            else:
                pulse_period, max_scale = 1.5, 1.08
                time_in_pulse_cycle = (t - pop_end_time) % pulse_period
                pulse_progress = time_in_pulse_cycle / pulse_period
                scale = 1 + (max_scale - 1) * np.sin(pulse_progress * np.pi)
            
            # Ensure scale doesn't make it too small or negative during pop-in if progress is 0
            current_scale = max(0.01, scale) # Prevent zero or negative size

            # Calculate new dimensions
            new_w_float = original_button_size[0] * current_scale
            new_h_float = original_button_size[1] * current_scale

            # Ensure dimensions are at least 1 after int conversion
            # This is the crucial change:
            final_w = max(1, int(new_w_float))
            final_h = max(1, int(new_h_float)) 
            return (final_w, final_h)
            
        button_clip_obj = button_clip_obj.resize(animate_button_size).set_position(('center', button_final_y))



        #### Arrow overlay

        arrows_overlay = mp.VideoFileClip("arrows_2_3.mov",has_mask=True)
        st.write("Duration:", arrows_overlay.duration)
        st.write("Has mask?", arrows_overlay.mask is not None)
        # arrows_overlay = arrows_overlay.set_mask(
        # arrows_overlay.mask.fx(lambda m: m.to_ImageClip().fl_image(lambda img: (img > 0.95).astype(float))))
        final_arrow_y = int(0.46 * resolution[1])
        final_arrow_x = int(0.275 * resolution[0])
        # arrows_overlay = arrows_overlay.rotate(-90, apply_to='mask')
        arrows_overlay = arrows_overlay.loop(duration=duration).resize(0.67)

        arrows_overlay = arrows_overlay.set_position((final_arrow_x, final_arrow_y )).set_start(5.5)

        # arrows_overlay = (
        #                     mp.VideoFileClip("arrows_2.mov", has_mask=True)
        #                     .rotate(-90)
        #                     .resize(width=0.07 * 1280)
        #                     .loop(duration=duration)
        #                     .set_position(("center", int(0.78 * resolution[1])))
        #                     .set_start(3))
        #                      # .set_mask(lambda: arrows_overlay.mask.fl_image(lambda img: (img > 0.95).astype(float)))


        final_clip = mp.CompositeVideoClip(
            [background_final, text_clip1_obj, text_clip2_obj, text_clip3_obj, button_clip_obj,arrows_overlay],
            size=resolution
        ).set_duration(duration)
        
        st.write("MoviePy: Visuals composed.")
        return final_clip

    except Exception as e:
        logging.error(f"Error in create_facebook_ad_new: {e}", exc_info=True)
        st.error(f"MoviePy Error during ad creation: {e}")
        return None # Fallback if error during creation
    finally:
        # It's good practice to close clips, though MoviePy often handles it.
        # For generated clips like ColorClip, ImageClip from array, TextClip, explicit close is less critical
        # than for file-based clips if they were opened and not processed into a final clip.
        if background_clip_obj and hasattr(background_clip_obj, 'close'): background_clip_obj.close()
        # Clips returned by rounded_bg_text are CompositeVideoClips; their components are managed.
        # if text_clip1_obj and hasattr(text_clip1_obj, 'close'): text_clip1_obj.close() # ... and others
        # if button_clip_obj and hasattr(button_clip_obj, 'close'): button_clip_obj.close()
        pass


# --- S3 Upload Function for Video ---
def upload_video_file_to_s3(
    file_path: str,
    bucket_name: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    object_name: str = None,
    region_name: str = 'us-east-1',
    content_type: str = 'video/mp4'
) -> str | None:
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id.strip(),
            aws_secret_access_key=aws_secret_access_key.strip(),
            region_name=region_name.strip()
        )

        if not object_name:
            timestamp = int(time.time())
            random_suffix = random.randint(1000, 9999)
            original_filename = os.path.basename(file_path)
            name, ext = os.path.splitext(original_filename)
            safe_name = "".join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in name)
            object_name = f"videos/{safe_name}_{timestamp}_{random_suffix}{ext}"

        with open(file_path, "rb") as f:
            s3_client.upload_fileobj(f, bucket_name.strip(), object_name, ExtraArgs={'ContentType': content_type})
        url = f"https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}"
        logging.info(f"Video uploaded to S3: {url}")
        return url
    except Exception as e:
        logging.error(f"Error uploading video {file_path} to S3: {e}", exc_info=True)
        st.error(f"S3 Upload Error for {os.path.basename(file_path)}: {e}")
        return None

# --- Main Video Generation Pipeline for a Single Video ---
def generate_single_video(
    video_topic: str,
    language: str,
    voice_id: str,
    openai_client: OpenAI,
    anthropic_api_key: str,
    s3_config: dict
):
    logging.info(f"--- Starting video generation for topic: '{video_topic}', lang: '{language}', voice: '{voice_id}' ---")
    st.info(f"Processing: {video_topic} ({language}, voice: {voice_id})")
    
    temp_video_file_path = None
    temp_bg_image_path = None
    tts_audio_file_path_local = None
    video_s3_url = None
    
    # MoviePy clips to close
    video_visuals_clip_obj = None
    tts_audio_clip_obj = None
    final_video_clip_to_write_obj = None

    try:
        # 1. Generate Image Prompt (using Claude)
        image_prompt_generation_prompt = f"write engaging image prompt for " + video_topic + "make sure to extract the visual aspect of the topic, that can convice people to click and show the positive most direct benefit (negate any non tangible aspect of the topic) make it look really good and attractive. ideally show a real life scenario people can relate.. like for 'bank repossessed cars' show a lot of cars in a lot and people around them. no overlay text on image!!! NO  TEXT ON IMAGE!!!"

#         image_prompt_generation_prompt = (
#     f"""Craft a SINGLE, vivid image-generation prompt for the topic: “{video_topic}”.

# Your goal: an irresistible thumbnail that **stops the scroll** and sparks immediate curiosity.

# 🏆  Must-have ingredients
# 1. **Big visual payoff** – show the *tangible* benefit or “after” moment in action.
# 2. **Human hook** – include at least one real person with a clear facial emotion  
#    (amazement, satisfaction, discovery) pointing, gazing, or reacting to the scene.
# 3. **Tension & reveal** – frame the shot so the subject feels *mid-action* or partially
#    hidden, hinting there’s more to see if the viewer clicks.
# 4. **Photorealistic, candid** – smartphone-style authenticity, natural lighting, slight imperfections.
# 5. **Color pop** – one strong accent color (clothing, object, sign) that draws the eye.
# 6. **NO text, watermarks, logos, filters, AI artifacts, or studio lighting.**

# End your prompt with these exact tags (for the diffusion model):
# “photorealistic, candid, unstaged, natural lighting, dynamic composition, shallow depth of field, no text on image, no logos, no watermark.”
# """
# )

        image_prompt_for_fal = generate_text_with_claude(
            prompt=image_prompt_generation_prompt,
            model = "claude-opus-4-20250514",
            anthropic_api_key=anthropic_api_key
        ) + "\n looks great\n photorealistic, candid unstaged"
        if not image_prompt_for_fal:
            st.warning(f"Could not generate image prompt for '{video_topic}'. Using topic as fallback.")
            image_prompt_for_fal = f"{video_topic}, \n looks great\n photorealistic, candid unstaged" # Basic fallback

        # 2. Generate and Download Background Image (Fal)
        fal_image_info = generate_fal_image(full_prompt=image_prompt_for_fal)
        bg_image_for_video_path = None
        if fal_image_info and 'url' in fal_image_info:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img_file:
                temp_bg_image_path = tmp_img_file.name
            if download_image(fal_image_info['url'], temp_bg_image_path):
                bg_image_for_video_path = temp_bg_image_path
                st.image(bg_image_for_video_path)
            else: # Download failed
                if os.path.exists(temp_bg_image_path): os.remove(temp_bg_image_path)
                temp_bg_image_path = None # Ensure it's None so fallback is used
        # If bg_image_for_video_path is None, create_facebook_ad_new handles fallback

        # 3. Generate Narration Script Text using Claude (with language)
        narration_prompt = f"In {language}, Create a short, , and engaging narration script (about 1-2 sentences, around 8-10 seconds read time) for a Facebook video ad. dont use these or simillar: 'today' or 'limted time' 'x% off discount' or 'Apply Now' 'instant' 'in 1 minute' , dont use 'our' or 'we'.  for topic : {video_topic} ,.The narration should complement this, be encouraging, and invite viewers to learn more (use somethink like ...'Click now to ...'). dont make huge out there bombastic promises or too sensetional or use in the style of 'get approved' 'apply here' 'browse items...'  or make up info BUT still make people click and be cliffhangry. Ensure the script is entirely in {language}. \n  End with a strong convinsing  CTA in the likes of: 'Click to explore options or 'Tap to see how it works.' dont use 'to see models\what's available ... etc'   \nagain, do be sesetional, just a bit, and dont make up promises not provided as input" 
        narration_script_text = generate_text_with_claude(
            prompt=narration_prompt,
            anthropic_api_key=anthropic_api_key
        )

        # 4. Generate TTS audio (OpenAI) 
        if narration_script_text:
            tts_audio_file_path_local = generate_audio_with_timestamps(
                text=narration_script_text, openai_client=openai_client, voice_id=voice_id
            )
            if not tts_audio_file_path_local:
                st.warning(f"Failed to generate TTS for '{video_topic}' ({language}). Video will be silent.")
        else:
            st.warning(f"No narration script for '{video_topic}' ({language}). Video will be silent.")

        # 5. Generate Video Captions (Claude, with language)
        caption_prompt = f"""write a json with text to be shown as caption on video (the captions complete a sentence togther) not overly promising! dont make up info, for topic article about {video_topic} in language:{language} , must be 3 captions , each 2 words, for high ctr in like this format, not over sensetional and dont make big promises! : """ + """{'caption1' : 'BAD CREDIT?' ,'caption2' : 'RV OWNERSHIP' ,'caption3' : 'STILL POSSIBLE!'  }
                                         return JUST the json
"""
        captions_json_str = generate_text_with_claude(
            prompt=caption_prompt, anthropic_api_key=anthropic_api_key,model = "claude-3-7-sonnet-latest" # Haiku is good for structured JSON
        )
        captions_data = {}
        if captions_json_str:
            try:
                json_start = captions_json_str.find('{')
                json_end = captions_json_str.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    captions_data = json.loads(captions_json_str[json_start:json_end])
                else: raise ValueError("No JSON object delimiters found.")
            except Exception as e: # Catches JSONDecodeError and ValueError
                logging.error(f"Failed to parse/find captions JSON for '{video_topic}' ({language}): {e}. Response: {captions_json_str}")
                st.warning(f"Could not parse captions for '{video_topic}' ({language}). Using defaults.")
        
        default_captions_map = {
            "English": {"c1": "INTERESTED?", "c2": video_topic[:15].upper(), "c3": "SEE HOW!"},
            "Spanish": {"c1": "¿INTERESADO?", "c2": video_topic[:15].upper(), "c3": "¡DESCUBRE!"},
            # Add more languages as needed
        }
        default_lang_captions = default_captions_map.get(language, default_captions_map["English"])
        headline_text1 = captions_data.get("caption1", default_lang_captions["c1"])
        headline_text2 = captions_data.get("caption2", default_lang_captions["c2"])
        headline_text3 = captions_data.get("caption3", default_lang_captions["c3"])



        learn_more_text = generate_text_with_claude(f"""write 'Learn More Now' in {language}, return just the text1!!!""" ,anthropic_api_key=anthropic_api_key, model="claude-3-7-sonnet-20250219" ).replace("'","").replace('"',"")

        # Determine video duration
        video_duration_final = 7  # Default if no audio
        if tts_audio_file_path_local and os.path.exists(tts_audio_file_path_local):
            try:
                audio_clip_for_duration = mp.AudioFileClip(tts_audio_file_path_local)
                video_duration_final = max(5.0, min(30.0, audio_clip_for_duration.duration + 0.5)) # Add buffer, ensure min/max
                audio_clip_for_duration.close()
            except Exception as e:
                logging.warning(f"Could not get audio duration for '{video_topic}': {e}. Using default {video_duration_final}s.")
                st.warning(f"MoviePy: Could not read audio duration for {video_topic}. Using default.")
        
        # 6. Create Video Visuals (MoviePy)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file_obj:
            temp_video_file_path = temp_video_file_obj.name
        # temp_video_file_obj is closed here, path is retained

        video_visuals_clip_obj = create_facebook_ad_new(
            bg_img_path=bg_image_for_video_path, # Can be None
            headline_text1=headline_text1, headline_text2=headline_text2, headline_text3=headline_text3,
            duration=video_duration_final, learn_more=learn_more_text
        )

        if not video_visuals_clip_obj:
            st.error(f"MoviePy: Failed to create video visuals for {video_topic}. Skipping this video.")
            raise Exception("Visuals creation failed.") # Propagate to finally for cleanup

        # 7. Combine Video Visuals with TTS Audio
        final_video_clip_to_write_obj = video_visuals_clip_obj # Default to visuals only
        if tts_audio_file_path_local and os.path.exists(tts_audio_file_path_local):
            try:
                tts_audio_clip_obj = mp.AudioFileClip(tts_audio_file_path_local)
                # Match video duration to audio clip's duration
                final_video_clip_to_write_obj = video_visuals_clip_obj.set_duration(tts_audio_clip_obj.duration).set_audio(tts_audio_clip_obj)
                logging.info(f"TTS audio track added to video '{video_topic}'. Duration: {tts_audio_clip_obj.duration:.2f}s.")
            except Exception as e:
                logging.error(f"Error adding audio to video '{video_topic}': {e}. Proceeding with visuals only (if possible).")
                st.warning(f"MoviePy: Error adding audio for {video_topic}. Video might be silent or use visual's duration.")
                # Visuals_clip already has a duration, so it can be used as is.
        else: # No audio or audio failed
             final_video_clip_to_write_obj = video_visuals_clip_obj.set_duration(video_duration_final)


        # 8. Write the final video to a temporary local file
        st.write(f"MoviePy: Writing video file for '{video_topic}'...")
        final_video_clip_to_write_obj.write_videofile(
            temp_video_file_path, fps=30, codec='libx264', audio_codec='aac',
            preset='medium', threads=4, logger=None # 'bar' or None
        )
        st.write(f"MoviePy: Video file for '{video_topic}' written locally.")

        # 9. Upload video to S3
        if os.path.exists(temp_video_file_path) and os.path.getsize(temp_video_file_path) > 0:
            s3_object_name = f"videos/{video_topic.replace(' ','_').lower()}_{language.lower()}_{int(time.time())}_{random.randint(100,999)}.mp4"
            video_s3_url = upload_video_file_to_s3(
                file_path=temp_video_file_path,
                bucket_name=s3_config["bucket_name"],
                aws_access_key_id=s3_config["access_key_id"],
                aws_secret_access_key=s3_config["secret_access_key"],
                region_name=s3_config["region_name"],
                object_name=s3_object_name
            )
        else:
            st.error(f"Temporary video file for '{video_topic}' not found or empty. Cannot upload.")

    except Exception as e:
        logging.error(f"Error in generate_single_video for '{video_topic}', {language}: {e}", exc_info=True)
        st.error(f"Overall failure for '{video_topic}' ({language}): {e}")
    finally:
        # Cleanup MoviePy clips
        if video_visuals_clip_obj: video_visuals_clip_obj.close()
        if tts_audio_clip_obj: tts_audio_clip_obj.close()
        if final_video_clip_to_write_obj: final_video_clip_to_write_obj.close()
        
        # Cleanup temporary files
        for f_path in [temp_bg_image_path, tts_audio_file_path_local, temp_video_file_path]:
            if f_path and os.path.exists(f_path):
                try: 
                    os.remove(f_path)
                    logging.info(f"Cleaned up temp file: {f_path}")
                except Exception as e_del: 
                    logging.warning(f"Could not delete temp file {f_path}: {e_del}")
        logging.info(f"--- Finished video generation attempt for topic: '{video_topic}', lang: '{language}' ---")
    return video_s3_url


# --- Streamlit App UI ---
def run_streamlit_app():
    st.title("Video Slide")
    # st.markdown("Automate the creation of short video ads with AI-generated content and visuals.")

    # Sidebar for configurations
 
   
        

    st.header("📤 Input Video Tasks")
    st.markdown("""
    **CSV Format:** Columns: `topic` (text), `count` (number), `language` (e.g., English, Spanish), `voice` (OpenAI voice ID like 'alloy', 'nova', or custom like 'sage').
    """)

    input_source = st.radio("Choose input method:", ( "Enter Manually","Upload CSV"), horizontal=True)
    
    input_df = None

    if input_source == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            try:
                input_df = pd.read_csv(uploaded_file)
                required_cols = ['topic', 'count', 'language', 'voice']
                if not all(col in input_df.columns for col in required_cols):
                    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
                    input_df = None # Invalidate df
                else:
                    # Basic type conversion and validation
                    input_df['topic'] = input_df['topic'].astype(str)
                    input_df['language'] = input_df['language'].astype(str)
                    input_df['voice'] = input_df['voice'].astype(str)
                    input_df['count'] = pd.to_numeric(input_df['count'], errors='coerce').fillna(1).astype(int)
                    input_df = input_df[input_df['count'] > 0] # Filter out zero counts
                    st.dataframe(input_df.head())
            except Exception as e:
                st.error(f"Error reading or processing CSV: {e}")
                input_df = None
    else: # Manual Input
        if 'manual_df' not in st.session_state:
            st.session_state.manual_df = pd.DataFrame([
                {"topic": "Eco-Friendly Homes","language": "English", "count": 1,  "voice": "sage"}
                
            ])
        
        st.subheader("Manually Add/Edit Topics")
        edited_df = st.data_editor(
            st.session_state.manual_df,
            num_rows="dynamic",
            key="data_editor",
            column_config={
                "topic": st.column_config.TextColumn("Topic", required=True, width="large"),
                "count": st.column_config.NumberColumn("Count", min_value=1, max_value=5, step=1, required=True),
                "language": st.column_config.TextColumn("Language", required=True, help="E.g., English, Spanish, French, Korean"),
                "voice": st.column_config.SelectboxColumn(
                    "Voice", 
                    options=["alloy", "echo", "fable", "onyx", "nova", "shimmer", # OpenAI standard
                             "sage", "redneck", "announcer", "announcer uk"], # Custom mapped
                    required=True
                )
            }
        )
        st.session_state.manual_df = edited_df
        input_df = edited_df.copy()


    if st.button("🚀 Generate Videos", type="primary", disabled=(input_df is None or input_df.empty)):
        # Retrieve latest values from session state
        current_openai_api_key = openai_api_key
        current_anthropic_api_key = anthropic_api_key
        s3_config = {
            "bucket_name": s3_bucket_name,
            "access_key_id": aws_access_key,
            "secret_access_key": aws_secret_key,
            "region_name": s3_region
        }

        # if not all([current_openai_api_key, current_anthropic_api_key, s3_config["bucket_name"], s3_config["access_key_id"], s3_config["secret_access_key"]]):
        #     st.error("🚨 Please provide all API keys and S3 configuration details in the sidebar.")
        #     return

        if input_df is None or input_df.empty:
            st.warning("⚠️ No valid topics to process. Please upload a CSV or add topics manually.")
            return
        
        # Final validation of manual input df (drop fully empty rows if any)
        input_df.dropna(subset=['topic', 'language', 'voice'], how='all', inplace=True)
        input_df = input_df[input_df['topic'].astype(str).str.strip() != ''] # Remove rows with empty topic
        input_df['count'] = pd.to_numeric(input_df['count'], errors='coerce').fillna(1).astype(int)
        input_df = input_df[input_df['count'] > 0]
        
        if input_df.empty:
            st.warning("⚠️ No valid topics after final cleanup. Please check your input.")
            return

        try:
            openai_client = OpenAI(api_key=current_openai_api_key)
            # Test Anthropic client init (optional, generate_text_with_claude does it)
            # anthropic.Anthropic(api_key=current_anthropic_api_key).count_tokens("test")
        except Exception as e:
            st.error(f"🚨 Failed to initialize API client(s): {e}")
            return
        
        output_data_rows = []
        max_videos_for_any_topic = 0
        
        total_videos_to_attempt = input_df['count'].sum()
        progress_bar_st = st.progress(0.0, text="Initializing video generation...")
        videos_completed_count = 0

        st.markdown("---")
        status_expander = st.expander("📊 Generation Status & Logs", expanded=True)
        
        with status_expander:
            for index, row in input_df.iterrows():
                topic_val = str(row['topic'])
                count_val = int(row['count'])
                lang_val = str(row['language'])
                voice_val = str(row['voice'])

                st.markdown(f"Processing: **{topic_val}** ({lang_val}) - {count_val} video(s) with voice '{voice_val}'")
                
                video_urls_for_current_row = []
                for i in range(count_val):
                    current_video_num_text = f"Video {i+1}/{count_val} for '{topic_val}'"
                    st.caption(f"Starting {current_video_num_text}...")
                    progress_val = (videos_completed_count / total_videos_to_attempt) if total_videos_to_attempt > 0 else 0
                    progress_bar_st.progress(progress_val, text=f"{current_video_num_text} ({int(progress_val*100)}%)")

                    video_url = generate_single_video(
                        video_topic=topic_val, language=lang_val, voice_id=voice_val,
                        openai_client=openai_client, anthropic_api_key=current_anthropic_api_key,
                        s3_config=s3_config
                    )
                    video_urls_for_current_row.append(video_url if video_url else "FAILED")
                    videos_completed_count += 1
                
                output_row_dict = {"Topic": topic_val, "Language": lang_val}
                for i_url, url in enumerate(video_urls_for_current_row):
                    output_row_dict[f"Video_{i_url+1}_URL"] = url
                
                output_data_rows.append(output_row_dict)
                if len(video_urls_for_current_row) > max_videos_for_any_topic:
                    max_videos_for_any_topic = len(video_urls_for_current_row)
            
            progress_bar_st.progress(1.0, text="All video generation tasks processed.")

        if output_data_rows:
            video_url_cols = [f"Video_{i+1}_URL" for i in range(max_videos_for_any_topic)]
            final_df_cols = ["Topic", "Language"] + video_url_cols
            
            normalized_output_data = []
            for row_d in output_data_rows:
                new_r = {"Topic": row_d["Topic"], "Language": row_d["Language"]}
                for col_n in video_url_cols:
                    new_r[col_n] = row_d.get(col_n, "") 
                normalized_output_data.append(new_r)

            results_df = pd.DataFrame(normalized_output_data, columns=final_df_cols)
            st.success("✅ Video generation process completed!")
            st.header("🎞️ Generated Video URLs")
            st.dataframe(results_df)

            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download results as CSV", data=csv_export,
                file_name='generated_video_links.csv', mime='text/csv',
            )
        else:
            st.warning("No videos were generated. Check inputs or logs in console.")
    
    st.markdown("---")

if __name__ == "__main__":
    run_streamlit_app()
