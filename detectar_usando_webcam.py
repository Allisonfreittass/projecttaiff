import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import tkinter as tk
from tkinter import font, filedialog, Toplevel
from PIL import Image, ImageTk

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Controle de Qualidade TAIFF")
        self.root.configure(bg="#282a36")

        self.title_font = font.Font(family="Helvetica", size=28, weight="bold")
        self.title_label = tk.Label(
            root,
            text="Controle de Qualidade TAIFF",
            font=self.title_font,
            bg="#44475a",
            fg="white",
            padx=20,
            pady=10,
            relief="groove"
        )
        self.title_label.pack(pady=(20, 10))

        self.button_frame = tk.Frame(root, bg="#282a36")
        self.button_frame.pack(pady=10)

        button_style = {
            'bg': '#6272a4',
            'fg': 'white',
            'font': ('Helvetica', 12),
            'width': 12,
            'relief': 'flat',
            'bd': 2
        }

        self.toggle_camera_button = tk.Button(
            self.button_frame,
            text="Ativar Câmera",
            command=self.toggle_camera,
            **button_style
        )
        self.toggle_camera_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.toggle_photo_button = tk.Button(
            self.button_frame,
            text="Adicionar Foto",
            command=self.toggle_photo,
            **button_style
        )
        self.toggle_photo_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.expand_photo_button = tk.Button(
            self.button_frame,
            text="Expandir Foto",
            command=self.expand_photo,
            state=tk.DISABLED,
            **button_style
        )
        self.expand_photo_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.canvas_width = 960
        self.canvas_height = 720
        self.canvas = tk.Canvas(
            root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack(pady=(10, 20), padx=20)

        self.model = YOLO("runs/detect/train6/weights/best.pt")
        self.track_history = defaultdict(list)
        self.seguir = True
        self.deixar_rastro = True
        self.camera_active = False
        self.photo_active = False
        self.update_task = None
        self.imgtk = None
        self.uploaded_img = None
        self.processed_img = None  

    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
            self.toggle_camera_button.config(text="Ativar Câmera", bg="#6272a4")
            self.canvas.config(bg="black")
        else:
            self.start_camera()
            self.toggle_camera_button.config(text="Desativar Câmera", bg="#50fa7b")
            self.canvas.config(bg="#282a36")

    def toggle_photo(self):
        if self.photo_active:
            self.close_photo()
            self.toggle_photo_button.config(text="Adicionar Foto", bg="#6272a4")
        else:
            self.upload_photo()
            self.toggle_photo_button.config(text="Remover Foto", bg="#ff5555")

    def expand_photo(self):
        if self.processed_img is not None:  
            top = Toplevel(self.root)
            top.attributes('-fullscreen', True) 
            top.title("Imagem em Tela Cheia")

            screen_width = top.winfo_screenwidth()
            screen_height = top.winfo_screenheight()

            img = cv2.cvtColor(self.processed_img, cv2.COLOR_BGR2RGB)

            height, width, _ = img.shape
            aspect_ratio = width / height

            if aspect_ratio > (screen_width / screen_height):
                new_width = screen_width
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = screen_height
                new_width = int(new_height * aspect_ratio)

            img = cv2.resize(img, (new_width, new_height))
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            img_label = tk.Label(top, image=imgtk)
            img_label.image = imgtk
            img_label.pack()

            top.bind("<Escape>", lambda e: top.destroy())  

    def start_camera(self):
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Erro ao abrir a câmera.")
                return
            self.camera_active = True
            self.update_frame()

    def stop_camera(self):
        if self.camera_active:
            self.camera_active = False
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.update_task is not None:
                self.root.after_cancel(self.update_task)
                self.update_task = None
            self.canvas.delete("all")

    def update_frame(self):
        if self.camera_active:
            success, img = self.cap.read()
            if success:
                height, width, _ = img.shape
                aspect_ratio = width / height
                if aspect_ratio > (self.canvas_width / self.canvas_height):
                    new_width = self.canvas_width
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = self.canvas_height
                    new_width = int(new_height * aspect_ratio)

                img = cv2.resize(img, (new_width, new_height))
                background = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
                y_offset = (self.canvas_height - new_height) // 2
                x_offset = (self.canvas_width - new_width) // 2
                background[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img

                img = background

                if self.seguir:
                    results = self.model.track(img, persist=True)
                else:
                    results = self.model(img)

                for result in results:
                    img = result.plot()
                    if self.seguir and self.deixar_rastro:
                        try:
                            boxes = result.boxes.xywh.cpu()
                            track_ids = result.boxes.id.int().cpu().tolist()
                            for box, track_id in zip(boxes, track_ids):
                                x, y, w, h = box
                                track = self.track_history[track_id]
                                track.append((float(x), float(y)))
                                if len(track) > 30:
                                    track.pop(0)
                                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                                cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                        except:
                            pass

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.root.imgtk = imgtk

            self.update_task = self.root.after(10, self.update_frame)

    def upload_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.uploaded_img = cv2.imread(file_path)
            img = cv2.resize(self.uploaded_img, (960, 720))
            results = self.model(img)
            img = results[0].plot() 
            self.processed_img = img  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.root.imgtk = imgtk
            self.photo_active = True
            self.expand_photo_button.config(state=tk.NORMAL)

    def close_photo(self):
        self.canvas.delete("all")
        self.photo_active = False
        self.expand_photo_button.config(state=tk.DISABLED)

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
