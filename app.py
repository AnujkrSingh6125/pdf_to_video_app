import streamlit as st
import os
import json
import asyncio
import tempfile
import pypdf
import docx
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from google import genai
from google.genai import types

# Page Config
st.set_page_config(page_title="PDF to AI Video Explainer", page_icon="🎬", layout="centered")

st.title("🎬 Document to AI Video Explainer")
st.markdown("Upload a PDF or Word document and generate an explainer video instantly!")

# 1. Fetch API Key Securely from Streamlit Secrets
# The app will look in the secrets vault for GEMINI_API_KEY
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("⚠️ API Key not configured. The developer needs to add GEMINI_API_KEY to Streamlit Secrets.")
    st.stop()

# File Uploader Widget
uploaded_file = st.file_uploader("Upload your document (.pdf or .docx)", type=["pdf", "docx"])

def extract_text(file_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    if ext == ".pdf":
        reader = pypdf.PdfReader(tmp_path)
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    elif ext == ".docx":
        doc = docx.Document(tmp_path)
        for p in doc.paragraphs:
            text += p.text + "\n"
            
    os.remove(tmp_path)
    return text.strip()

def get_script(text, api_key):
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Divide this document into a 3-scene video explanation script.
    Return strictly JSON with this schema:
    [
      {{
        "scene_number": 1,
        "slide_title": "Title Here",
        "bullet_points": ["Point 1", "Point 2"],
        "narration": "Voiceover narration script here."
      }}
    ]
    Content: {text[:4000]}
    """
    res = client.models.generate_content(
        model='gemini-1.5-flash',,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(res.text)

def make_slide(scene, img_path):
    img = Image.new("RGB", (1920, 1080), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    
    # Header Title
    draw.text((100, 120), scene["slide_title"].upper(), fill=(56, 189, 248))
    draw.line([(100, 180), (600, 180)], fill=(56, 189, 248), width=5)
    
    # Bullet Points
    y = 280
    for pt in scene["bullet_points"]:
        draw.text((120, y), f"• {pt}", fill=(241, 245, 249))
        y += 100
    img.save(img_path)

async def make_audio(text, audio_path):
    comm = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await comm.save(audio_path)

# Processing Trigger
if uploaded_file:
    if st.button("🚀 Generate Video Explainer", type="primary"):
        status_box = st.empty()
        progress_bar = st.progress(0)
        
        try:
            status_box.info("📄 Extracting text from uploaded document...")
            text = extract_text(uploaded_file.getvalue(), uploaded_file.name)
            progress_bar.progress(20)
            
            status_box.info("🧠 Generating script with Gemini AI...")
            # Passes the securely loaded API key
            scenes = get_script(text, API_KEY)
            progress_bar.progress(40)
            
            clips = []
            temp_files = []
            
            for idx, scene in enumerate(scenes):
                status_box.info(f"🎨 Rendering Scene {idx+1}/{len(scenes)}...")
                img_p = f"temp_img_{idx}.png"
                aud_p = f"temp_aud_{idx}.mp3"
                temp_files.extend([img_p, aud_p])
                
                make_slide(scene, img_p)
                asyncio.run(make_audio(scene["narration"], aud_p))
                
                aud = AudioFileClip(aud_p)
                clip = ImageClip(img_p).set_duration(aud.duration).set_audio(aud)
                clips.append(clip)
            
            progress_bar.progress(80)
            status_box.info("🎞️ Stitching scenes into final MP4 video...")
            
            out_video = "final_output.mp4"
            final_video = concatenate_videoclips(clips, method="compose")
            final_video.write_videofile(out_video, fps=24)
            
            progress_bar.progress(100)
            status_box.success("🎉 Video generation complete!")
            
            # Integrated Video Player & Download Button
            st.video(out_video)
            with open(out_video, "rb") as f:
                st.download_button("📥 Download Video MP4", data=f, file_name="explainer.mp4", mime="video/mp4")
                
            # Temporary file cleanup
            for f in temp_files + [out_video]:
                if os.path.exists(f): os.remove(f)
                
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
