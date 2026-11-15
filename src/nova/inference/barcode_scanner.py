import cv2
import numpy as np
from pyzbar.pyzbar import decode

def scan_barcode_from_image(uploaded_file):
    """
    Đọc mã vạch từ file ảnh tải lên qua Streamlit.
    Trả về chuỗi mã vạch (barcode string) nếu tìm thấy, ngược lại trả về None.
    """
    try:
        # Chuyển file ảnh tải lên thành mảng numpy để OpenCV đọc
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Giải mã vạch bằng pyzbar
        decoded_objects = decode(opencv_image)
        
        for obj in decoded_objects:
            barcode_data = obj.data.decode('utf-8')
            barcode_type = obj.type
            print(f"Đã tìm thấy mã vạch ({barcode_type}): {barcode_data}")
            return barcode_data
            
        return None
    except Exception as e:
        print(f"Lỗi khi quét mã vạch: {e}")
        return None