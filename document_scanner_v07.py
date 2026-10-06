import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

# Попытка импорта tkinterdnd2
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    print("Библиотека tkinterdnd2 не установлена. Drag & drop будет недоступен.")

class DocumentScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Сканер документов")
        self.root.geometry("1200x700")

        # Список документов
        self.documents = []
        self.current_index = None

        # Текущий документ
        self.original_image = None
        self.processed_image = None
        self.clicked_points = []
        self.current_warped = None

        # Параметры улучшения
        self.brightness = tk.IntVar(value=0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.black_white = tk.BooleanVar(value=False)

        # Перетаскивание углов
        self.dragging_corner = None

        # Масштаб отображения
        self.display_scale = 1.0
        self.display_max_width = 550
        self.display_max_height = 450

        self.create_widgets()

        # Включение drag & drop, если доступно
        if HAS_DND:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)

    def create_widgets(self):
        # Основной layout
        main_panel = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5)
        main_panel.pack(fill=tk.BOTH, expand=True)

        # Левая панель: список документов
        left_frame = tk.Frame(main_panel, width=250)
        main_panel.add(left_frame, minsize=200)

        tk.Label(left_frame, text="Документы:").pack(pady=5)

        btn_add = tk.Button(left_frame, text="Добавить фото", command=self.add_images)
        btn_add.pack(fill=tk.X, padx=5, pady=2)

        btn_remove = tk.Button(left_frame, text="Удалить выбранное", command=self.remove_current)
        btn_remove.pack(fill=tk.X, padx=5, pady=2)

        # Кнопки перемещения
        move_frame = tk.Frame(left_frame)
        move_frame.pack(fill=tk.X, padx=5, pady=2)
        btn_up = tk.Button(move_frame, text="▲ Вверх", command=self.move_up)
        btn_up.pack(side=tk.LEFT, expand=True, fill=tk.X)
        btn_down = tk.Button(move_frame, text="▼ Вниз", command=self.move_down)
        btn_down.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Список документов
        list_frame = tk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        self.listbox.bind('<<ListboxSelect>>', self.on_select_document)

        # Кнопка очистки внизу, отдельно и заметно
        btn_clear = tk.Button(left_frame, text="Очистить все", command=self.clear_all,
                              bg='#ffdddd', activebackground='#ffaaaa')
        btn_clear.pack(fill=tk.X, padx=5, pady=(15, 5))

        # Правая панель
        right_frame = tk.Frame(main_panel)
        main_panel.add(right_frame)

        toolbar = tk.Frame(right_frame, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_zoom_out = tk.Button(toolbar, text="−", width=2, command=self.zoom_out)
        btn_zoom_out.pack(side=tk.LEFT, padx=(10,0))

        btn_zoom_in = tk.Button(toolbar, text="+", width=2, command=self.zoom_in)
        btn_zoom_in.pack(side=tk.LEFT, padx=2)

        btn_save_img = tk.Button(toolbar, text="Сохранить как изображение", command=self.save_image)
        btn_save_img.pack(side=tk.LEFT, padx=10, pady=5)

        btn_save_pdf = tk.Button(toolbar, text="Сохранить все в PDF", command=self.save_all_pdf)
        btn_save_pdf.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Frame(toolbar, width=2, bd=1, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)

        tk.Label(toolbar, text="Яркость:").pack(side=tk.LEFT, padx=(10,0))
        scale_brightness = tk.Scale(toolbar, from_=-100, to=100, orient=tk.HORIZONTAL,
                                    variable=self.brightness, length=150,
                                    command=lambda _: self.apply_enhancements())
        scale_brightness.pack(side=tk.LEFT, padx=5)

        tk.Label(toolbar, text="Контраст:").pack(side=tk.LEFT, padx=(10,0))
        scale_contrast = tk.Scale(toolbar, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                                  variable=self.contrast, length=150,
                                  command=lambda _: self.apply_enhancements())
        scale_contrast.pack(side=tk.LEFT, padx=5)

        chk_bw = tk.Checkbutton(toolbar, text="Чёрно-белый", variable=self.black_white,
                                command=self.apply_enhancements)
        chk_bw.pack(side=tk.LEFT, padx=10)

        display_frame = tk.Frame(right_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.lbl_original_title = tk.Label(display_frame, text="Исходное изображение")
        self.lbl_original_title.grid(row=0, column=0, sticky='w', padx=5, pady=2)

        self.canvas_original = tk.Canvas(display_frame, bg='gray', highlightthickness=0)
        self.canvas_original.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')

        self.lbl_processed_title = tk.Label(display_frame, text="Обработанное изображение")
        self.lbl_processed_title.grid(row=0, column=1, sticky='w', padx=5, pady=2)

        self.lbl_processed = tk.Label(display_frame, bg='gray')
        self.lbl_processed.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

        display_frame.grid_columnconfigure(0, weight=1, uniform='group')
        display_frame.grid_columnconfigure(1, weight=1, uniform='group')
        display_frame.grid_rowconfigure(1, weight=1)

        self.canvas_original.bind("<ButtonPress-1>", self.on_corner_press)
        self.canvas_original.bind("<B1-Motion>", self.on_corner_drag)
        self.canvas_original.bind("<ButtonRelease-1>", self.on_corner_release)
        self.canvas_original.bind("<MouseWheel>", self.on_mousewheel)

    # ==================== Drag & Drop ====================
    def on_drop(self, event):
        if not HAS_DND:
            return
        files = self.root.tk.splitlist(event.data)
        self.load_images_from_paths(files)

    def load_images_from_paths(self, paths):
        for path in paths:
            ext = path.lower().split('.')[-1]
            if ext not in ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'):
                continue
            try:
                with open(path, 'rb') as f:
                    data = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                h, w = img.shape[:2]
                points = [(int(w*0.05), int(h*0.05)),
                          (int(w*0.95), int(h*0.05)),
                          (int(w*0.95), int(h*0.95)),
                          (int(w*0.05), int(h*0.95))]
                doc = {
                    'original': img,
                    'processed': img.copy(),
                    'points': points,
                    'warped': None
                }
                self.documents.append(doc)
                self.listbox.insert(tk.END, f"Документ {len(self.documents)}")
            except Exception:
                continue

        if self.documents and self.current_index is None:
            self.select_document(0)

    # ==================== Управление документами ====================
    def add_images(self):
        file_paths = filedialog.askopenfilenames(
            title="Выберите фото документов",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("Все файлы", "*.*")]
        )
        if file_paths:
            self.load_images_from_paths(file_paths)

    def remove_current(self):
        if self.current_index is None:
            return
        idx = self.current_index
        self.documents.pop(idx)
        self.listbox.delete(idx)
        if self.documents:
            new_idx = min(idx, len(self.documents)-1)
            self.select_document(new_idx)
        else:
            self.clear_display()
            self.current_index = None

    def clear_all(self):
        self.documents.clear()
        self.listbox.delete(0, tk.END)
        self.clear_display()
        self.current_index = None

    def move_up(self):
        """Переместить выбранный документ на одну позицию вверх."""
        if self.current_index is None or self.current_index == 0:
            return
        idx = self.current_index
        # Меняем местами в списке документов
        self.documents[idx], self.documents[idx-1] = self.documents[idx-1], self.documents[idx]
        # Обновляем Listbox
        text = self.listbox.get(idx)
        self.listbox.delete(idx)
        self.listbox.insert(idx-1, text)
        # Обновляем current_index
        self.current_index = idx - 1
        self.select_document(self.current_index)

    def move_down(self):
        """Переместить выбранный документ на одну позицию вниз."""
        if self.current_index is None or self.current_index >= len(self.documents)-1:
            return
        idx = self.current_index
        # Меняем местами в списке документов
        self.documents[idx], self.documents[idx+1] = self.documents[idx+1], self.documents[idx]
        # Обновляем Listbox
        text = self.listbox.get(idx)
        self.listbox.delete(idx)
        self.listbox.insert(idx+1, text)
        # Обновляем current_index
        self.current_index = idx + 1
        self.select_document(self.current_index)

    def clear_display(self):
        self.original_image = None
        self.processed_image = None
        self.clicked_points = []
        self.current_warped = None
        self.canvas_original.delete("all")
        self.lbl_processed.configure(image='')
        self.lbl_processed.image = None

    def select_document(self, index):
        if index < 0 or index >= len(self.documents):
            return
        self.current_index = index
        doc = self.documents[index]
        self.original_image = doc['original']
        self.processed_image = doc['processed']
        self.clicked_points = doc['points'].copy()
        self.current_warped = doc['warped']
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(index)
        self.listbox.activate(index)
        self.update_original_canvas()
        self.update_processed_display()

    def on_select_document(self, event):
        selection = self.listbox.curselection()
        if selection:
            self.select_document(selection[0])

    # ==================== Перетаскивание углов ====================
    def on_corner_press(self, event):
        if self.original_image is None or not self.clicked_points:
            return
        min_dist = 15
        closest = None
        for i, (x_orig, y_orig) in enumerate(self.clicked_points):
            dx = x_orig / self.display_scale
            dy = y_orig / self.display_scale
            dist = ((event.x - dx)**2 + (event.y - dy)**2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest = i
        if closest is not None:
            self.dragging_corner = closest

    def on_corner_drag(self, event):
        if self.dragging_corner is None or self.original_image is None:
            return
        x_orig = int(event.x * self.display_scale)
        y_orig = int(event.y * self.display_scale)
        img_h, img_w = self.original_image.shape[:2]
        x_orig = max(0, min(x_orig, img_w-1))
        y_orig = max(0, min(y_orig, img_h-1))
        self.clicked_points[self.dragging_corner] = (x_orig, y_orig)
        if self.current_index is not None:
            self.documents[self.current_index]['points'] = self.clicked_points.copy()
        self.update_original_canvas()

    def on_corner_release(self, event):
        if self.dragging_corner is None:
            return
        self.dragging_corner = None
        self.perform_manual_warp()

    # ==================== Применение перспективы и улучшений ====================
    def perform_manual_warp(self):
        if len(self.clicked_points) != 4:
            return
        pts = np.array(self.clicked_points, dtype="float32")
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self.original_image, M, (maxWidth, maxHeight))

        self.current_warped = warped
        if self.current_index is not None:
            self.documents[self.current_index]['warped'] = warped
        self.apply_enhancements()

    def apply_enhancements(self):
        if self.original_image is None:
            return
        base_img = self.current_warped if self.current_warped is not None else self.original_image
        enhanced = self.enhance_image(base_img)
        self.processed_image = enhanced
        if self.current_index is not None:
            self.documents[self.current_index]['processed'] = enhanced
        self.update_processed_display()

    def enhance_image(self, img):
        result = img.astype(np.float32)
        contrast = self.contrast.get()
        result = (result - 127) * contrast + 127
        brightness = self.brightness.get()
        result += brightness
        result = np.clip(result, 0, 255).astype(np.uint8)

        if self.black_white.get():
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        return result

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    # ==================== Отображение ====================
    def zoom_in(self):
        if self.display_max_width < 1200:
            self.display_max_width += 100
            self.display_max_height += 80
            self.refresh_displays()

    def zoom_out(self):
        if self.display_max_width > 200:
            self.display_max_width -= 100
            self.display_max_height -= 80
            self.refresh_displays()

    def on_mousewheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def refresh_displays(self):
        self.update_original_canvas()
        self.update_processed_display()

    def update_original_canvas(self):
        if self.original_image is None:
            self.canvas_original.delete("all")
            return

        img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        pil_img.thumbnail((self.display_max_width, self.display_max_height), Image.Resampling.LANCZOS)
        self.display_scale = self.original_image.shape[1] / pil_img.width

        if self.clicked_points:
            draw = ImageDraw.Draw(pil_img)
            pts_display = [(x / self.display_scale, y / self.display_scale) for x, y in self.clicked_points]
            for i in range(4):
                p1 = pts_display[i]
                p2 = pts_display[(i + 1) % 4]
                draw.line([p1, p2], fill='yellow', width=3)
            for i, (dx, dy) in enumerate(pts_display):
                r = 8
                draw.ellipse((dx-r, dy-r, dx+r, dy+r), fill='red', outline='yellow', width=2)
                draw.text((dx+12, dy-12), str(i+1), fill='red')

        self.display_original = ImageTk.PhotoImage(pil_img)
        self.canvas_original.delete("all")
        self.canvas_original.create_image(0, 0, anchor=tk.NW, image=self.display_original)
        self.canvas_original.config(width=pil_img.width, height=pil_img.height)

    def update_processed_display(self):
        if self.processed_image is None:
            self.lbl_processed.configure(image='')
            self.lbl_processed.image = None
            return
        img_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((self.display_max_width, self.display_max_height), Image.Resampling.LANCZOS)
        self.display_processed = ImageTk.PhotoImage(pil_img)
        self.lbl_processed.configure(image=self.display_processed)
        self.lbl_processed.image = self.display_processed

    # ==================== Сохранение ====================
    def save_image(self):
        if self.processed_image is None:
            messagebox.showwarning("Предупреждение", "Нет обработанного изображения.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tiff")]
        )
        if not file_path:
            return
        cv2.imwrite(file_path, self.processed_image)
        messagebox.showinfo("Сохранено", f"Изображение сохранено:\n{file_path}")

    def save_all_pdf(self):
        if not self.documents:
            messagebox.showwarning("Предупреждение", "Нет документов для сохранения.")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")]
        )
        if not file_path:
            return

        pil_images = []
        for doc in self.documents:
            img = doc['processed'] if doc['processed'] is not None else doc['original']
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_images.append(Image.fromarray(img_rgb))

        try:
            if pil_images:
                pil_images[0].save(file_path, "PDF", resolution=100.0,
                                   save_all=True, append_images=pil_images[1:])
            messagebox.showinfo("Сохранено", f"Многостраничный PDF сохранён:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить PDF:\n{e}")

if __name__ == "__main__":
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = DocumentScannerApp(root)
    root.mainloop()