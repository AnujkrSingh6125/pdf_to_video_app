import os
import json
import asyncio
import tempfile
import streamlit as st
import google.generativeai as genai

# Document Parsers
import pypdf
import docx

# Video & Audio Generation
import edge_tts
from moviepy.editor import TextClip, AudioFileClip, CompositeVideoClip, ColorClip

# =========================================================
# 1. HELPER FUNCTIONS & GEMINI API CALL
# =========================================================

def extract_text(uploaded_file):
    """Extracts raw text from PDF or DOCX files."""
    text = ""
    file_type = uploaded_file.name.split(".")[-1].lower()
    
    if file_type == "pdf":
        reader = pypdf.PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    elif file_type in ["docx", "doc"]:
        doc = docx.Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
            
    return text.strip()


def get_script(text, api_key):
    """Generates a structured 3-scene video script using Gemini API."""
    genai.configure(api_key=api_key)
    
    # Using the standard gemini-1.5-flash model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    You are an expert video creator. Divide this document content into a concise 3-scene video script.
    Return strictly a JSON array without markdown formatting, using this exact schema:
    [
      {{
        "scene_number": 1,
        "slide_title": "Title for Scene 1",
        "bullet_points": ["Key Point 1", "Key Point 2"],
        "narration": "Voiceover narration script for scene 1."
      }},
      {{
        "scene_number": 2,
        "slide_title": "Title for Scene 2",
        "bullet_points": ["Key Point 1", "Key Point 2"],
        "narration": "Voiceover narration script for scene 2."
      }},
      {{
        "scene_number": 3,
        "slide_title": "Title for Scene 3",
        "bullet_points": ["Key Point 1", "Key Point 2"],
        "narration": "Voiceover narration script for scene 3."
      }}
    ]

    Document Content:
    {text[:4000]}
    """
    
    res = model.generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    return json.loads(res.text)


async def generate_audio(text, output_path):
    """Generates speech audio from text using edge-tts."""
    communicate = edge_tts.Communicate(text, "en-US-ChristopherNeural")
    await communicate.save(output_path)


def create_video(script_data):
    """Builds a MP4 video file from the generated script JSON."""
    temp_dir = tempfile.mkdtemp()
    scene_clips = []
    
    for idx, scene in enumerate(script_data):
        # 1. Generate TTS Audio
        audio_path = os.path.join(temp_dir, f"scene_{idx}.mp3")
        asyncio.run(generate_audio(scene["narration"], audio_path))
        
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        
        # 2. Create Background Clip (1280x720 Dark Theme)
        bg_clip = ColorClip(size=(1280, 720), color=(30, 30, 45), duration=duration)
        
        # 3. Create Text Overlay
        slide_text = f"{scene['slide_title']}\n\n" + "\n".join([f"• {bp}" for bp in scene["bullet_points"]])
        
        txt_clip = TextClip(
            slide_text,
            fontsize=36,
            color='white',
            font='Arial',
            method='caption',
            size=(1100, 600)
        ).set_position('center').set_duration(duration)
        
        # 4. Combine Audio + Visuals
        composite = CompositeVideoClip([bg_clip, txt_clip]).set_audio(audio_clip)
        scene_clips.append(composite)
    
    # Concatenate all scene clips together
    from moviepy.editor import concatenate_videoclips
    final_clip = concatenate_videoclips(scene_clips, method="compose")
    
    output_video_path = os.path.join(temp_dir, "final_explainer.mp4")
    final_clip.write_videofile(
        output_video_path,
        fps=24,
        codec="libx264",
        audio_codec="aac"
    )
    
    return output_video_path


# =========================================================
# 2. STREAMLIT USER INTERFACE
# =========================================================

st.set_page_config(page_title="PDF to AI Video Explainer", layout="centered")

st.title("📄 PDF to AI Video Explainer")
st.write("Upload a PDF or Word document and generate an explainer video instantly!")

uploaded_file = st.file_uploader("Upload your document (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file is not None:
    if st.button("🚀 Generate Video Explainer"):
        try:
            # Check for API Key in Secrets
            if "GEMINI_API_KEY" not in st.secrets:
                st.error("Missing GEMINI_API_KEY in Streamlit Secrets. Please configure it in your Streamlit Cloud settings.")
                st.stop()
                
            api_key = st.secrets["GEMINI_API_KEY"]
            
            # Step 1: Extract Text
            with st.spinner("📖 Extracting text from document..."):
                extracted_text = extract_text(uploaded_file)
                if not extracted_text:
                    st.error("Could not extract any readable text from this file.")
                    st.stop()
            
            # Step 2: Generate Script
            with st.spinner("🧠 Generating script with Gemini AI..."):
                script_data = get_script(extracted_text, api_key)
            
            # Step 3: Render Video
            with st.spinner("🎬 Generating voiceover and rendering video clips..."):
                video_file_path = create_video(script_data)
            
            st.success("🎉 Video generated successfully!")
            
            # Display Video
            st.video(video_file_path)
            
            # Download Button
            with open(video_file_path, "rb") as file:
                st.download_button(
                    label="📥 Download Explainer Video",
                    data=file,
                    file_name="explainer_video.mp4",
                    mime="video/mp4"
                )
                
        except Exception as e:
            st.error(f"Error during generation: {e}")
