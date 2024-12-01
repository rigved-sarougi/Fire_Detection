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

st.title('Fire Extinguisher')
st.sidebar.title('App Mode')

app_mode = st.sidebar.selectbox('Choose the App Mode',
                                ['About App', 'Run on Image', 'Run on Video', 'Missing Detection'])

if app_mode == 'About App':
    st.subheader("About")
    st.markdown("<h5>This is the Fire Extinguisher App created with custom trained models using YoloV5</h5>", unsafe_allow_html=True)
    st.markdown("""
                ## Features
- Detect on Image
- Detect on Videos
- Fire Extinguisher Missing Detection
## Tech Stack
- Python
- PyTorch
- Python CV
- Streamlit
- YoloV5
## 🔗 Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rigved-sarougi)
[![Github](https://img.shields.io/badge/Github-1DA1F2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Rigvedsarougi)
""")

if app_mode in ['Run on Image', 'Missing Detection']:
    st.subheader("Detected Fire Extinguishers:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    img_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if img_file:
        image = np.array(Image.open(img_file))
    else:
        image = np.array(Image.open(demo_img))
        
    st.sidebar.markdown("**Original Image**")
    st.sidebar.image(image)
    
    # Predict the image
    model = load_model()
    results = model(image)
    detections = results.xyxy[0]  # Detected bounding boxes
    output = np.squeeze(results.render())
    st.subheader("Output Image")
    st.image(output, use_column_width=True)
    
    # Missing detection logic
    if app_mode == 'Missing Detection':
        st.subheader("Missing Fire Extinguisher Detection:")
        
        # Define regions of interest (mock regions)
        predefined_regions = [(100, 50, 200, 150), (300, 100, 400, 200)]  # Example regions (x1, y1, x2, y2)
        detected_regions = [(int(x[0]), int(x[1]), int(x[2]), int(x[3])) for x in detections.tolist()]
        
        missing_regions = []
        for region in predefined_regions:
            if not any([is_overlap(region, detected) for detected in detected_regions]):
                missing_regions.append(region)
        
        # Highlight missing regions on the image
        for region in missing_regions:
            x1, y1, x2, y2 = region
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        st.image(output, caption="Highlighted Missing Locations", use_column_width=True)
        st.write(f"Number of missing fire extinguishers: {len(missing_regions)}")

if app_mode == 'Run on Video':
    st.subheader("Detected Fire Extinguishers:")
    text = st.markdown("")
    
    st.sidebar.markdown("---")
    st.subheader("Output")
    stframe = st.empty()
    
    video_file = st.sidebar.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi', 'asf', 'm4v'])
    st.sidebar.markdown("---")
    tffile = tempfile.NamedTemporaryFile(delete=False)
    
    if not video_file:
        vid = cv2.VideoCapture(demo_video)
        tffile.name = demo_video
    else:
        tffile.write(video_file.read())
        vid = cv2.VideoCapture(tffile.name)
    
    st.sidebar.markdown("**Input Video**")
    st.sidebar.video(tffile.name)
    
    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model = load_model()
        results = model(frame)
        output = np.squeeze(results.render())
        stframe.image(output)
