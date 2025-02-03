import streamlit as st
import torch
import base64
import time
import numpy as np
import pandas as pd
from object_detection import process_video
from monocular import MonocularDepth
from pathlib import Path
from model import get_probabilities  # Importing function that returns 50 probability values


def generate_video_html(video_path, caption):
    """Generate an HTML block to embed a video in Streamlit."""
    try:
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()
            video_base64 = base64.b64encode(video_data).decode("utf-8")
        video_html = f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3 style="color: #4CAF50; font-size: 18px; font-weight: bold;">{caption}</h3>
            <video width="100%" height="450px" autoplay loop muted 
                   style="border: 3px solid #4CAF50; border-radius: 12px; box-shadow: 0px 4px 10px rgba(0,0,0,0.3);">
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        """
        return video_html
    except Exception as e:
        st.error(f"Error generating video HTML: {e}")
        return ""


def main():
    st.set_page_config(page_title="Accident Anticipation", layout="wide")

    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stTitle {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .stSubtitle {
            text-align: center;
            font-size: 20px;
            color: #666;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown("<h1 class='stTitle'>🚗 Accident Anticipation Visualization</h1>", unsafe_allow_html=True)
    st.markdown("<p class='stSubtitle'>Analyze Dashcam Footage for Object Detection & Depth Estimation</p>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov", "mkv"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        sample_dir = Path("./sample")
        sample_dir.mkdir(exist_ok=True)
        input_video_path = sample_dir / uploaded_file.name
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.read())

        filename = input_video_path.stem
        object_video_path = sample_dir / f"{filename}_object.mp4"
        depth_video_path = sample_dir / f"{filename}_depth.mp4"

        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            analyze = st.button("🔍 Analyze Video", use_container_width=True)

        if analyze:
            with st.spinner("🚧 Processing video for object detection, please wait..."):
                # process_video(str(input_video_path))
                pass

            with st.spinner("🔍 Generating depth estimation, please wait..."):
                # depth_model = MonocularDepth(device="cuda" if torch.cuda.is_available() else "cpu")
                # depth_model.video_depth(str(input_video_path), str(sample_dir), colourize=True)
                pass

            st.success("🎉 Video processing complete!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(generate_video_html(str(input_video_path), "Original Video"), unsafe_allow_html=True)
            with col2:
                st.markdown(generate_video_html(str(object_video_path), "Object Detection"), unsafe_allow_html=True)
            with col3:
                st.markdown(generate_video_html(str(depth_video_path), "Depth Estimation"), unsafe_allow_html=True)

            st.markdown("## 🔍 Live Probability Plot")

            chart_placeholder = st.empty()

            frame_nums = list(range(1, 51))
            prob_values = [0] * 50

            for frame in range(50):
                prob_values[frame] = get_probabilities(frame)
                df = pd.DataFrame({"Frame": frame_nums[:frame + 1], "Probability": prob_values[:frame + 1]})
                chart_placeholder.line_chart(df.set_index("Frame"))
                time.sleep(0.1)

            if max(prob_values) > 0.5:
                st.markdown(
                    """
                    <div style="padding: 15px; background-color: #ffcccc; border-radius: 8px; text-align: center;">
                        <h4 style="color: #d9534f;">⚠️ High Probability Detected! Take necessary precautions.</h4>
                    </div>
                    """, unsafe_allow_html=True
                )


if __name__ == "__main__":
    main()