import streamlit as st
import cv2
from ultralytics import YOLO
import pytesseract
from PIL import Image
import re
import numpy as np

# Cấu hình pytesseract cho Tiếng Việt
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Đường dẫn đến tesseract
custom_config = r'--oem 3 --psm 6 -l vie'  # Chế độ Tesseract và ngôn ngữ là Tiếng Việt

# Load mô hình YOLOv8 đã được huấn luyện
model = YOLO('./best.pt')

# Hàm để làm sạch văn bản trích xuất
def clean_text(text):
    # Loại bỏ các ký tự không mong muốn
    text = re.sub(r'[^\w\s@.-]', '', text)
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    return text

# Hàm để nhận diện và trích xuất thông tin
def detect_and_extract(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Chuyển ảnh grayscale thành ảnh 3 kênh
    gray_image_3_channel = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    results = model(gray_image_3_channel)
    detected_info = {'name': [], 'phone': [], 'email': [], 'address': []}
    
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box tọa độ
        labels = result.boxes.cls.cpu().numpy()  # Nhãn của từng bounding box
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = map(int, box[:4])
            roi = image[y1:y2, x1:x2]

            text = pytesseract.image_to_string(roi, config=custom_config).strip()
            text = clean_text(text)  # Làm sạch văn bản trích xuất

            # Gán văn bản vào đúng loại thông tin dựa trên nhãn
            if label == 0: 
                detected_info['name'].append(text)
            elif label == 1:  
                detected_info['phone'].append(text)
            elif label == 2: 
                detected_info['email'].append(text)
            elif label == 3: 
                detected_info['address'].append(text)

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return image, detected_info
# Streamlit app
st.title("trich xuat danh thiep")

uploaded_file = st.file_uploader("Chon 1 anh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption='tai len anh.', use_column_width=True)
    st.write("")
    st.write("Detecting...")

    detected_image, detected_info = detect_and_extract(image)

    st.image(detected_image, caption='Detected Image.', use_column_width=True)
    st.write("thong tin trich xuat:")
    st.write("Name: " + ", ".join(detected_info['name']))
    st.write("Phone: " + ", ".join(detected_info['phone']))
    st.write("Email: " + ", ".join(detected_info['email']))
    st.write("Address: " + ", ".join(detected_info['address']))