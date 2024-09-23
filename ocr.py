import cv2
from ultralytics import YOLO
import pytesseract

# Cấu hình pytesseract cho Tiếng Việt
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # Đường dẫn đến tesseract
custom_config = r'--oem 3 --psm 6 -l vie'  # Chế độ Tesseract và ngôn ngữ là Tiếng Việt

# Load mô hình YOLOv8 đã được huấn luyện
model = YOLO('./best.pt')

# Load ảnh danh thiếp
image = cv2.imread('./test_image1.jpg')

results = model(image)

detected_info = {'name': [], 'phone': [], 'email': [], 'address': []}

# Vẽ bounding boxes trên ảnh và trích xuất thông tin
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box tọa độ
    labels = result.boxes.cls.cpu().numpy()  # Nhãn của từng bounding box
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box[:4])
        roi = image[y1:y2, x1:x2]

        # Sử dụng Tesseract OCR để trích xuất văn bản từ khu vực này
        text = pytesseract.image_to_string(roi, config=custom_config).strip()

        # Gán văn bản vào đúng loại thông tin dựa trên nhãn
        if label == 0: 
            detected_info['name'].append(text)
        elif label == 1:  
            detected_info['phone'].append(text)
        elif label == 2: 
            detected_info['email'].append(text)
        elif label == 3:  
            detected_info['address'].append(text)

        # Vẽ khung và thêm text lên ảnh
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# In ra kết quả theo định dạng yêu cầu
print("name:", ", ".join(detected_info['name']))
print("phones:", ", ".join(detected_info['phone']))
print("email:", ", ".join(detected_info['email']))
print("address:", ", ".join(detected_info['address']))

# Thay đổi kích thước ảnh về chiều rộng 640 pixel
h, w, _ = image.shape
new_width = 640
new_height = int(h * (new_width / w))  # Tính toán chiều cao tương ứng để giữ tỉ lệ
resized_image = cv2.resize(image, (new_width, new_height))

# Hiển thị ảnh với kích thước 640 pixel
cv2.imshow('trich xuat danh thiep', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()