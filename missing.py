import streamlit as st
import cv2
import numpy as np
import av
import torch
import tempfile
from PIL import Image

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="weights/last.pt", force_reload=True)
    return model

demo_img = "Fire-Extinguisher-Safety.jpg"
demo_video = "Fire-Extinguisher.mp4"

st.title('Fire Extinguisher Detector')
st.sidebar.title('App Mode')

app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App', 'Run on Image', 'Run on Video'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown("<h5>This app detects fire extinguishers and identifies missing ones using YOLOv5.</h5>", unsafe_allow_html=True)
    st.markdown("""
                ## Features
- Detect Fire Extinguishers in Images
- Detect Fire Extinguishers in Videos
- Identify Missing Fire Extinguishers
## Tech Stack
- Python
- PyTorch
- OpenCV
- Streamlit
- YOLOv5
## 🔗 Links
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rigved-sarougi)
[![GitHub](https://img.shields.io/badge/GitHub-1DA1F2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Rigvedsarougi)
""")

if app_mode == 'Run on Image':
    st.subheader("Detect Fire Extinguishers")
    text = st.markdown("")
    
    # Input for Image
    img_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = np.array(Image.open(img_file))
    else:
        image = np.array(Image.open(demo_img))
        
    st.sidebar.markdown("---")
    st.sidebar.image(image, caption="Original Image", use_column_width=True)
    
    # Predict the image
    model = load_model()
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()
    output = np.squeeze(results.render())
    detected_count = len(detections)
    
    # Define zones for expected fire extinguishers (e.g., predefined coordinates)
    expected_zones = [
        {"zone": [100, 200, 300, 400], "label": "Zone 1"},
        {"zone": [400, 200, 600, 400], "label": "Zone 2"}
    ]
    
    # Check for missing extinguishers
    missing_zones = []
    for zone in expected_zones:
        x1, y1, x2, y2 = zone["zone"]
        found = any((det[0] >= x1 and det[1] >= y1 and det[2] <= x2 and det[3] <= y2) for det in detections)
        if not found:
            missing_zones.append(zone["label"])
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Mark missing zones in blue

    # Display results
    text.write(f"<h1 style='text-align: center; color:red;'>Detected: {detected_count}</h1>", unsafe_allow_html=True)
    if missing_zones:
        st.error(f"Missing fire extinguishers in: {', '.join(missing_zones)}")
    else:
        st.success("All fire extinguishers are in place!")
    
    st.image(output, caption="Output Image", use_column_width=True)

if app_mode == 'Run on Video':
    st.subheader("Detect Fire Extinguishers in Video")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    stframe = st.empty()
    
    # Input for Video
    video_file = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    tffile = tempfile.NamedTemporaryFile(delete=False)
    
    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)
    
    st.sidebar.video(tffile.name)
    
    model = load_model()
    
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)
        detections = results.xyxy[0].cpu().numpy()
        output = np.squeeze(results.render())
        detected_count = len(detections)
        
        # Check for missing extinguishers (similar to image logic)
        missing_zones = []
        for zone in expected_zones:
            x1, y1, x2, y2 = zone["zone"]
            found = any((det[0] >= x1 and det[1] >= y1 and det[2] <= x2 and det[3] <= y2) for det in detections)
            if not found:
                missing_zones.append(zone["label"])
                cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Mark missing zones in blue
        
        # Display results
        text.write(f"<h1 style='text-align: center; color:red;'>Detected: {detected_count}</h1>", unsafe_allow_html=True)
        if missing_zones:
            st.error(f"Missing fire extinguishers in: {', '.join(missing_zones)}")
        else:
            st.success("All fire extinguishers are in place!")
        
        stframe.image(output)
