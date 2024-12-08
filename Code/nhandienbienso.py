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
                if w / h > 1 and w / h < 5:
                    license_plate = original_image[y:y+h, x:x+w]
                    break
    return original_image, license_plate

def sort_characters(contours):
    
    bounding_boxes = [cv2.boundingRect(cntr) for cntr in contours]
    def group_by_row(boxes, threshold=20):
        rows = []
        for box in sorted(boxes, key=lambda b: b[1]):
            placed = False
            for row in rows:
                if abs(row[-1][1] - box[1]) < threshold:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])
        return rows

    rows = group_by_row(bounding_boxes)

    for row in rows:
        row.sort(key=lambda b: b[0])
    sorted_boxes = [box for row in rows for box in row]

    sorted_contours = [contours[bounding_boxes.index(box)] for box in sorted_boxes]
    return sorted_contours

def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cntrs:
        return np.array([])
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]

    lower_width, upper_width, lower_height, upper_height = dimensions

    valid_contours = []
    for cntr in cntrs:
        x, y, w, h = cv2.boundingRect(cntr)
        if lower_width < w < upper_width and lower_height < h < upper_height:
            valid_contours.append(cntr)
            
    if not valid_contours:
        return np.array([])
    img_copy = img.copy()
    cv2.imshow("Filtered Contours", img_copy)
    cv2.waitKey(1)

    sorted_contours = sort_characters(valid_contours)

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
    
    img_gray_lp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray_lp = cv2.GaussianBlur(img_gray_lp,(5,5),0)

    _, img_binary_lp = cv2.threshold(img_gray_lp, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    dimensions = [LP_WIDTH / 12, LP_WIDTH / 4, LP_HEIGHT / 4, 3 * LP_HEIGHT / 5]

    img_binary_lp[0:5, :] = 255 
    img_binary_lp[:, 0:5] = 255 
    img_binary_lp[LP_WIDTH-10:LP_WIDTH, :] = 255 
    img_binary_lp[:, LP_HEIGHT-5:LP_HEIGHT] = 255  

    char_list = find_contours(dimensions, img_binary_lp)

    if char_list.size == 0:
        print("Không tìm thấy ký tự nào!")
        return [], img_binary_lp

    return char_list, img_binary_lp

model = load_model("character_recognition_model.h5")
characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" 

def preprocess_character(char_image):
    char_image = cv2.resize(char_image, (28, 28))
    char_image = char_image.astype("float32") / 255.0
    char_image = np.expand_dims(char_image, axis=-1)
    char_image = np.expand_dims(char_image, axis=0)  
    return char_image

def recognize_characters(segmented_characters):
    recognized_text = ""
    for char_image in segmented_characters:
        preprocessed = preprocess_character(char_image)  
        prediction = model.predict(preprocessed)  
        char_index = np.argmax(prediction) 
        recognized_text += characters[char_index] 
    return recognized_text

captured_license_plate = None

def capture_image():
    global captured_license_plate

    if captured_license_plate is None:
        print("Không có ảnh biển số để xử lý!")
        return

    segmented_characters, license_plate_with_contours = segment_characters(captured_license_plate)

    if len(segmented_characters) > 0:
        recognized_text = recognize_characters(segmented_characters)

        for widget in character_frame.winfo_children():
            widget.destroy()
        for char_img in segmented_characters:
            char_rgb = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)
            pil_char = Image.fromarray(char_rgb)
            pil_char.thumbnail((50, 50))
            tk_char = ImageTk.PhotoImage(pil_char)
            char_label = tk.Label(character_frame, image=tk_char)
            char_label.image = tk_char
            char_label.pack(side="left", padx=5)

        for widget in result_frame.winfo_children():
            widget.destroy()
        result_label = tk.Label(result_frame, text=recognized_text, font=("Helvetica", 16), fg="#007BFF", bg="white")
        result_label.pack(fill="both", expand=True)
    else:
        print("Không tìm thấy ký tự nào để phân đoạn!")
        for widget in result_frame.winfo_children():
            widget.destroy()
        error_label = tk.Label(result_frame, text="Không tìm thấy ký tự!", font=("Helvetica", 16), fg="red", bg="white")
        error_label.pack(fill="both", expand=True)




def start_camera():
    
    global input_img_label, output_img_label, captured_license_plate
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Không thể mở camera!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được khung hình!")
            break
        detected_frame, license_plate = detect_license_plate(frame)

        detected_frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(detected_frame_rgb)
        pil_frame.thumbnail((400, 400))
        tk_frame = ImageTk.PhotoImage(pil_frame)
        input_img_label.config(image=tk_frame)
        input_img_label.image = tk_frame
        
        if license_plate is not None:
            captured_license_plate = license_plate
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

        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
root.title("Nhận Diện Biển Số Xe - Camera")
root.geometry("800x620")
root.configure(bg="#f2f2f2")

title_label = tk.Label(root, text="HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE TỪ CAMERA", font=("Helvetica", 18, "bold"), bg="#4CAF50", fg="white")
title_label.pack(fill="x", pady=10)


input_title_label = tk.Label(root, text="Ảnh đầu vào:", font=("Helvetica", 12, "bold"), bg="#f2f2f2", fg="#333")
input_title_label.place(x=50, y=45)
input_frame = tk.Frame(root, width=420, height=280, relief="solid", borderwidth=1, bg="white")
input_frame.pack_propagate(False)
input_frame.place(x=50, y=70)
input_img_label = tk.Label(input_frame, text="Khung hình từ camera", bg="white", fg="#666", font=("Helvetica", 10, "italic"))
input_img_label.pack(fill="both", expand=True)

input_title_label = tk.Label(root, text="Biển số xe sau khi cắt:", font=("Helvetica", 12, "bold"), bg="#f2f2f2", fg="#333")
input_title_label.place(x=50, y=355) 
output_frame = tk.Frame(root, width=200, height=150, relief="solid", borderwidth=1, bg="white")
output_frame.pack_propagate(False)
output_frame.place(x=50, y=380)
output_img_label = tk.Label(output_frame, text="Biển số", bg="white", fg="#666", font=("Helvetica", 10, "italic"))
output_img_label.pack(fill="both", expand=True)

input_title_label = tk.Label(root, text="Ký tự phân đoạn::", font=("Helvetica", 12, "bold"), bg="#f2f2f2", fg="#333")
input_title_label.place(x=300, y=355) 
character_frame = tk.Frame(root, width=400, height=150, relief="solid", borderwidth=1, bg="white")
character_frame.pack_propagate(False)
character_frame.place(x=300, y=380)

input_title_label = tk.Label(root, text="Biển số xe:", font=("Helvetica", 12, "bold"), bg="#f2f2f2", fg="#333")
input_title_label.place(x=50, y=540) 
result_frame = tk.Frame(root, width=550, height=50, relief="solid", borderwidth=1, bg="white")
result_frame.pack_propagate(False)
result_frame.place(x=150, y=540)

btn_start_camera = tk.Button(root, text="Bắt đầu Camera", command=start_camera, font=("Helvetica", 12), bg="#007BFF", fg="white", activebackground="#0056b3", activeforeground="white")
btn_start_camera.place(x=550, y=70, width=150, height=50)

btn_capture_image = tk.Button(root, text="Lấy Ảnh", command=capture_image, font=("Helvetica", 12), bg="#FF5722", fg="white", activebackground="#E64A19", activeforeground="white")
btn_capture_image.place(x=550, y=140, width=150, height=50)

btn_exit = tk.Button(root, text="Thoát", command=root.quit, font=("Helvetica", 12), bg="#DC3545", fg="white", activebackground="#c82333", activeforeground="white")
btn_exit.place(x=550, y=210, width=150, height=50)

root.mainloop()
