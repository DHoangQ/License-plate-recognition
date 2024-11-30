import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model



def detect_license_plate(frame):
    original_image = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    license_plate = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w / h > 1 and w / h < 5:  # Tỷ lệ của biển số thường là 2-5
                    license_plate = original_image[y:y+h+10, x:x+w+10]
                    break
    return original_image, license_plate


#============================================================================================================================


def sort_characters(contours):
    # Tìm bounding box cho từng contour
    bounding_boxes = [cv2.boundingRect(cntr) for cntr in contours]
    
    # Phân nhóm các ký tự theo hàng
    def group_by_row(boxes, threshold=20):
        rows = []
        for box in sorted(boxes, key=lambda b: b[1]):  # Sắp xếp theo y (tọa độ dọc)
            placed = False
            for row in rows:
                if abs(row[-1][1] - box[1]) < threshold:  # Nếu cùng hàng (chênh lệch y nhỏ)
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])
        return rows

    # Phân nhóm bounding boxes theo hàng
    rows = group_by_row(bounding_boxes)
    
    # Sắp xếp trong từng hàng theo trục x
    for row in rows:
        row.sort(key=lambda b: b[0])  # Sắp xếp theo tọa độ x (trái sang phải)
    
    # Ghép tất cả các bounding boxes theo thứ tự hàng
    sorted_boxes = [box for row in rows for box in row]
    
    # Trả về contours đã được sắp xếp
    sorted_contours = [contours[bounding_boxes.index(box)] for box in sorted_boxes]
    return sorted_contours



def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cntrs:
        return np.array([])  # Trả về mảng rỗng nếu không có contours
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    # Lấy các giá trị giới hạn kích thước ký tự
    lower_width, upper_width, lower_height, upper_height = dimensions

    valid_contours = []
    for cntr in cntrs:
        x, y, w, h = cv2.boundingRect(cntr)
        if lower_width < w < upper_width and lower_height < h < upper_height:
            valid_contours.append(cntr)
            
    if not valid_contours:
        return np.array([])
    
    # Hiển thị các contour đã lọc (nếu cần)
    img_copy = img.copy()
    # cv2.drawContours(img_copy, valid_contours, 1, (255, 0, 0), 2)
    cv2.imshow("Filtered Contours", img_copy)
    cv2.waitKey(1)

    # Sắp xếp các contours theo thứ tự hàng trên trước, hàng dưới sau
    sorted_contours = sort_characters(valid_contours)

    # Xử lý tiếp tục như bình thường (tách ký tự, chuẩn hóa, v.v.)
    img_res = []
    for cntr in sorted_contours:
        x, y, w, h = cv2.boundingRect(cntr)
        char = img[y:y + h, x:x + w]
        char = cv2.resize(char, (20, 40))
        char_copy = np.zeros((44, 24), dtype=np.uint8)
        char = cv2.subtract(255, char)
        char_copy[2:42, 2:22] = char
        char_copy[0:2, :] = 0
        char_copy[:, 0:2] = 0
        char_copy[42:44, :] = 0
        char_copy[:, 22:24] = 0
        img_res.append(char_copy)

    return np.array(img_res)




def segment_characters(image):
    """Phân đoạn ký tự từ biển số."""
    # Tăng độ tương phản bằng cách làm mịn và giảm nhiễu
    img_gray_lp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray_lp = cv2.medianBlur(img_gray_lp, 5)

    # Chuyển đổi ảnh sang nhị phân
    _, img_binary_lp = cv2.threshold(img_gray_lp, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    # img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    # Kích thước biển số
    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Điều chỉnh thông số dimensions
    dimensions = [LP_WIDTH / 15, LP_WIDTH / 4, LP_HEIGHT / 4, 3 * LP_HEIGHT / 4]

    # Tìm và lọc các contours
    char_list = find_contours(dimensions, img_binary_lp)

    if char_list.size == 0:  # Kiểm tra nếu không tìm thấy ký tự
        print("Không tìm thấy ký tự nào!")
        return [], img_binary_lp

    return char_list, img_binary_lp


# Load mô hình nhận diện ký tự
model = load_model("character_recognition_model.h5")
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Các ký tự có thể nhận diện


def preprocess_character(char_image):
    """Chuẩn hóa ký tự để đưa vào mô hình dự đoán."""
    char_image = cv2.resize(char_image, (28, 28))  # Resize về kích thước đầu vào của mô hình
    char_image = char_image.astype("float32") / 255.0  # Chuẩn hóa giá trị pixel
    char_image = np.expand_dims(char_image, axis=-1)  # Thêm kênh (nếu cần thiết)
    char_image = np.expand_dims(char_image, axis=0)  # Thêm batch size
    return char_image

def recognize_characters(segmented_characters):
    """Nhận diện ký tự từ danh sách ký tự phân đoạn."""
    recognized_text = ""
    for char_image in segmented_characters:
        preprocessed = preprocess_character(char_image)  # Chuẩn hóa ảnh ký tự
        prediction = model.predict(preprocessed)  # Dự đoán
        char_index = np.argmax(prediction)  # Lấy chỉ số ký tự có xác suất cao nhất
        recognized_text += characters[char_index]  # Thêm ký tự vào chuỗi
    return recognized_text

# =================================================================================================

# Biến toàn cục lưu biển số đã cắt
captured_license_plate = None

# Hàm capture_image cập nhật
def capture_image():
    """Chụp ảnh biển số, phân đoạn và nhận diện ký tự."""
    global captured_license_plate

    if captured_license_plate is None:
        print("Không có ảnh biển số để xử lý!")
        return

    # Gọi hàm phân đoạn ký tự
    segmented_characters, license_plate_with_contours = segment_characters(captured_license_plate)

    if len(segmented_characters) > 0:
        # Nhận diện ký tự
        recognized_text = recognize_characters(segmented_characters)

        # Hiển thị kết quả nhận diện trong khung ký tự phân đoạn
        for widget in character_frame.winfo_children():
            widget.destroy()  # Xóa các ký tự cũ
        for char_img in segmented_characters:
            char_rgb = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)
            pil_char = Image.fromarray(char_rgb)
            pil_char.thumbnail((50, 50))
            tk_char = ImageTk.PhotoImage(pil_char)
            char_label = tk.Label(character_frame, image=tk_char)
            char_label.image = tk_char
            char_label.pack(side="left", padx=5)

        # Hiển thị chuỗi ký tự trong khung mới
        for widget in result_frame.winfo_children():
            widget.destroy()  # Xóa kết quả cũ
        result_label = tk.Label(result_frame, text=recognized_text, font=("Helvetica", 16), fg="#007BFF", bg="white")
        result_label.pack(fill="both", expand=True)
    else:
        print("Không tìm thấy ký tự nào để phân đoạn!")
        for widget in result_frame.winfo_children():
            widget.destroy()
        error_label = tk.Label(result_frame, text="Không tìm thấy ký tự!", font=("Helvetica", 16), fg="red", bg="white")
        error_label.pack(fill="both", expand=True)




def start_camera():
    """Hiển thị camera và nhận diện biển số."""
    global input_img_label, output_img_label, captured_license_plate
    cap = cv2.VideoCapture(0)  # Mở camera laptop (camera 0)

    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    while True:
        ret, frame = cap.read()  # Đọc khung hình từ camera
        if not ret:
            print("Không nhận được khung hình!")
            break

        # Nhận diện biển số
        detected_frame, license_plate = detect_license_plate(frame)

        # Hiển thị khung hình từ camera
        detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(detected_frame_rgb)
        pil_frame.thumbnail((400, 400))
        tk_frame = ImageTk.PhotoImage(pil_frame)
        input_img_label.config(image=tk_frame)
        input_img_label.image = tk_frame

        # Hiển thị biển số nếu phát hiện được
        if license_plate is not None:
            captured_license_plate = license_plate  # Lưu biển số đã cắt để xử lý sau
            print('da luu')
            license_rgb = cv2.cvtColor(license_plate, cv2.COLOR_BGR2RGB)
            pil_license = Image.fromarray(license_rgb)
            pil_license.thumbnail((200, 200))
            tk_license = ImageTk.PhotoImage(pil_license)
            output_img_label.config(image=tk_license, text="")
            output_img_label.image = tk_license
        else:
            output_img_label.config(text="Không tìm thấy biển số xe.", image=None)
            captured_license_plate = None

        # Cập nhật giao diện
        root.update()

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
# ========================================================================================================================================


# Tạo giao diện chính
root = tk.Tk()
root.title("Nhận Diện Biển Số Xe - Camera")
root.geometry("800x620")
root.configure(bg="#f2f2f2")

# Phần tiêu đề
title_label = tk.Label(root, text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE TỪ CAMERA", font=("Helvetica", 18, "bold"), bg="#4CAF50", fg="white")
title_label.pack(fill="x", pady=10)

# Khung hiển thị ảnh đầu vào
input_title_label = tk.Label(root, text="Ảnh đầu vào:", font=("Helvetica", 12, "bold"), bg="#f2f2f2", fg="#333")
input_title_label.place(x=50, y=45)  # Đặt vị trí bên trên khung input_frame
input_frame = tk.Frame(root, width=420, height=280, relief="solid", borderwidth=1, bg="white")
input_frame.pack_propagate(False)
input_frame.place(x=50, y=70)
input_img_label = tk.Label(input_frame, text="Khung hình từ camera", bg="white", fg="#666", font=("Helvetica", 10, "italic"))
input_img_label.pack(fill="both", expand=True)

# Khung hiển thị biển số cắt ra
input_title_label = tk.Label(root, text="Biển số xe sau khi cắt:", font=("Helvetica", 12, "bold"), bg="#f2f2f2", fg="#333")
input_title_label.place(x=50, y=355) 
output_frame = tk.Frame(root, width=200, height=150, relief="solid", borderwidth=1, bg="white")
output_frame.pack_propagate(False)
output_frame.place(x=50, y=380)
output_img_label = tk.Label(output_frame, text="Biển số", bg="white", fg="#666", font=("Helvetica", 10, "italic"))
output_img_label.pack(fill="both", expand=True)

# Khung hiển thị ký tự phân đoạn
input_title_label = tk.Label(root, text="Ký tự phân đoạn::", font=("Helvetica", 12, "bold"), bg="#f2f2f2", fg="#333")
input_title_label.place(x=300, y=355) 
character_frame = tk.Frame(root, width=400, height=150, relief="solid", borderwidth=1, bg="white")
character_frame.pack_propagate(False)
character_frame.place(x=300, y=380)

# Biển số đã đọc
input_title_label = tk.Label(root, text="Biển số xe:", font=("Helvetica", 12, "bold"), bg="#f2f2f2", fg="#333")
input_title_label.place(x=50, y=540) 
result_frame = tk.Frame(root, width=550, height=50, relief="solid", borderwidth=1, bg="white")
result_frame.pack_propagate(False)
result_frame.place(x=150, y=540)

# Nút bắt đầu camera
btn_start_camera = tk.Button(root, text="Bắt đầu Camera", command=start_camera, font=("Helvetica", 12), bg="#007BFF", fg="white", activebackground="#0056b3", activeforeground="white")
btn_start_camera.place(x=550, y=70, width=150, height=50)

# Nút lấy ảnh
btn_capture_image = tk.Button(root, text="Lấy Ảnh", command=capture_image, font=("Helvetica", 12), bg="#FF5722", fg="white", activebackground="#E64A19", activeforeground="white")
btn_capture_image.place(x=550, y=140, width=150, height=50)

# Nút Thoát
btn_exit = tk.Button(root, text="Thoát", command=root.quit, font=("Helvetica", 12), bg="#DC3545", fg="white", activebackground="#c82333", activeforeground="white")
btn_exit.place(x=550, y=210, width=150, height=50)

# Chạy ứng dụng
root.mainloop()
