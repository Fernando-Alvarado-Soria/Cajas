import csv
import json
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from tkinter import messagebox, ttk

import cv2

from detectar_caja import detect_objects, open_camera, parse_roi, resize_frame


class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de cajas")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.cap = None
        self.running = False
        self.last_clean_frame = None
        self.last_objects = []
        self.last_photo = None
        self.last_time = time.perf_counter()
        self.fps = 0.0
        self.captured_boxes = []
        self.placed_boxes = []
        self.unplaced_boxes = []
        self.capture_count = 0
        self.container_number = 1
        self.packing_started = False
        self.packing_dirty = False

        self.camera_var = tk.StringVar(value="1")
        self.width_var = tk.StringVar(value="640")
        self.height_var = tk.StringVar(value="480")
        self.mode_var = tk.StringVar(value="edge")
        self.roi_var = tk.StringVar(value="")
        self.min_area_var = tk.StringVar(value="1500")
        self.max_objects_var = tk.StringVar(value="10")
        self.min_aspect_var = tk.StringVar(value="0.35")
        self.max_aspect_var = tk.StringVar(value="7.0")
        self.min_extent_var = tk.StringVar(value="0.18")
        self.status_var = tk.StringVar(value="Camara detenida")
        self.container_status_var = tk.StringVar(value="Contenedor 1 | Capturas: 0 | Cajas: 0")
        self.container_length_var = tk.StringVar(value="60")
        self.container_width_var = tk.StringVar(value="12")
        self.container_height_var = tk.StringVar(value="13")
        self.px_per_cm_var = tk.StringVar(value="29.27")
        self.px_per_cm_width_var = tk.StringVar(value="38.24")
        self.reference_length_var = tk.StringVar(value="9.02")
        self.reference_width_var = tk.StringVar(value="5.1")

        self.build_layout()

    def build_layout(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(main, text="Controles", padding=10)
        controls.grid(row=0, column=0, sticky="ew")

        controls.columnconfigure(11, weight=1)

        self.add_input(controls, "Camara", self.camera_var, 0)
        self.add_input(controls, "Ancho", self.width_var, 1)
        self.add_input(controls, "Alto", self.height_var, 2)
        self.add_combo(controls, "Modo", self.mode_var, ["edge", "white"], 3)
        self.add_input(controls, "ROI", self.roi_var, 4, width=18)
        self.add_input(controls, "Area min", self.min_area_var, 5)

        ttk.Button(controls, text="Empezar deteccion", command=self.start_camera).grid(
            row=0, column=12, padx=5
        )
        ttk.Button(controls, text="Detener", command=self.stop_camera).grid(
            row=0, column=13, padx=5
        )
        ttk.Button(controls, text="Capturar", command=self.capture).grid(
            row=0, column=14, padx=5
        )

        advanced = ttk.Frame(controls)
        advanced.grid(row=1, column=0, columnspan=15, sticky="ew", pady=(8, 0))

        self.add_input(advanced, "Max detecciones", self.max_objects_var, 0)
        self.add_input(advanced, "Aspect min", self.min_aspect_var, 1)
        self.add_input(advanced, "Aspect max", self.max_aspect_var, 2)
        self.add_input(advanced, "Extent min", self.min_extent_var, 3)

        container_controls = ttk.LabelFrame(controls, text="Contenedor", padding=8)
        container_controls.grid(row=2, column=0, columnspan=15, sticky="ew", pady=(8, 0))

        self.add_input(container_controls, "Largo cm", self.container_length_var, 0)
        self.add_input(container_controls, "Ancho cm", self.container_width_var, 1)
        self.add_input(container_controls, "Alto cm", self.container_height_var, 2)
        self.add_input(container_controls, "Px/cm largo", self.px_per_cm_var, 3)
        self.add_input(container_controls, "Px/cm ancho", self.px_per_cm_width_var, 4)
        self.add_input(container_controls, "Ref largo", self.reference_length_var, 5)
        self.add_input(container_controls, "Ref ancho", self.reference_width_var, 6)
        ttk.Button(
            container_controls,
            text="Calibrar con deteccion",
            command=self.calibrate_from_detection,
        ).grid(row=0, column=7, padx=6, sticky="s")
        ttk.Button(
            container_controls,
            text="Actualizar contenedor",
            command=self.redraw_container,
        ).grid(row=0, column=8, padx=6, sticky="s")
        ttk.Button(
            container_controls,
            text="Empezar acomodo de contenedor",
            command=self.start_packing,
        ).grid(row=0, column=9, padx=6, sticky="s")
        ttk.Button(
            container_controls,
            text="Reacomodo de contenedor",
            command=self.repack_container,
        ).grid(row=0, column=10, padx=6, sticky="s")
        ttk.Button(
            container_controls,
            text="Limpiar acomodo",
            command=self.clear_packing,
        ).grid(row=0, column=11, padx=6, sticky="s")

        content = ttk.Frame(main)
        content.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0)
        content.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(content, anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        side = ttk.LabelFrame(content, text="Medidas detectadas", padding=8)
        side.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        columns = ("id", "x", "y", "w", "h", "rot", "area")
        self.tree = ttk.Treeview(side, columns=columns, show="headings", height=16)

        for column in columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=70, anchor="center")

        self.tree.grid(row=0, column=0, sticky="nsew")
        detections_scrollbar = ttk.Scrollbar(side, orient="vertical", command=self.tree.yview)
        detections_scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=detections_scrollbar.set)

        packing = ttk.LabelFrame(side, text="Acomodo del contenedor", padding=8)
        packing.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        self.container_canvas = tk.Canvas(
            packing,
            width=540,
            height=250,
            bg="#f4f4f4",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self.container_canvas.grid(row=0, column=0, sticky="ew")

        self.packing_summary_var = tk.StringVar(value="Sin cajas capturadas")
        ttk.Label(packing, textvariable=self.container_status_var, anchor="w").grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(6, 0),
        )
        ttk.Label(packing, textvariable=self.packing_summary_var, anchor="w").grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(6, 0),
        )

        box_columns = ("id", "largo", "ancho", "estado")
        self.box_tree = ttk.Treeview(
            packing,
            columns=box_columns,
            show="headings",
            height=8,
        )

        for column in box_columns:
            self.box_tree.heading(column, text=column)
            self.box_tree.column(column, width=95, anchor="center")

        self.box_tree.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        box_scrollbar = ttk.Scrollbar(packing, orient="vertical", command=self.box_tree.yview)
        box_scrollbar.grid(row=3, column=1, sticky="ns", pady=(8, 0))
        self.box_tree.configure(yscrollcommand=box_scrollbar.set)

        status = ttk.Label(main, textvariable=self.status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.redraw_container()

    def add_input(self, parent, label, variable, column, width=8):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, padx=4, sticky="w")
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=variable, width=width).grid(row=1, column=0)

    def add_combo(self, parent, label, variable, values, column):
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, padx=4, sticky="w")
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            frame,
            textvariable=variable,
            values=values,
            width=7,
            state="readonly",
        ).grid(row=1, column=0)

    def build_detector_args(self):
        return SimpleNamespace(
            mode=self.mode_var.get(),
            min_area=int(self.min_area_var.get()),
            max_objects=int(self.max_objects_var.get()),
            min_aspect=float(self.min_aspect_var.get()),
            max_aspect=float(self.max_aspect_var.get()),
            min_extent=float(self.min_extent_var.get()),
            allow_nested=False,
        )

    def get_camera_settings(self):
        return (
            int(self.camera_var.get()),
            int(self.width_var.get()),
            int(self.height_var.get()),
        )

    def get_container_settings(self):
        return (
            float(self.container_length_var.get()),
            float(self.container_width_var.get()),
            float(self.container_height_var.get()),
        )

    def get_px_per_cm(self):
        px_per_cm_length = float(self.px_per_cm_var.get())
        px_per_cm_width = float(self.px_per_cm_width_var.get())

        if px_per_cm_length <= 0 or px_per_cm_width <= 0:
            raise ValueError("Px/cm largo y Px/cm ancho deben ser mayores a 0.")

        return px_per_cm_length, px_per_cm_width

    def calibrate_from_detection(self):
        if not self.last_objects:
            messagebox.showinfo(
                "Sin deteccion",
                "Primero detecta una caja de referencia con medidas conocidas.",
            )
            return

        try:
            reference_length = float(self.reference_length_var.get())
            reference_width = float(self.reference_width_var.get())
        except ValueError:
            messagebox.showerror(
                "Referencia invalida",
                "Ref largo y Ref ancho deben ser numeros.",
            )
            return

        if reference_length <= 0 or reference_width <= 0:
            messagebox.showerror(
                "Referencia invalida",
                "Ref largo y Ref ancho deben ser mayores a 0.",
            )
            return

        reference_object = self.last_objects[0]
        px_per_cm_length = reference_object["rotated_w"] / reference_length
        px_per_cm_width = reference_object["rotated_h"] / reference_width
        self.px_per_cm_var.set(f"{px_per_cm_length:.2f}")
        self.px_per_cm_width_var.set(f"{px_per_cm_width:.2f}")
        self.recalculate_captured_box_sizes()

        messagebox.showinfo(
            "Calibracion actualizada",
            "Escala calculada con el objeto detectado 1:\n"
            f"Px/cm largo: {px_per_cm_length:.2f}\n"
            f"Px/cm ancho: {px_per_cm_width:.2f}\n\n"
            "Las cajas capturadas se recalcularon con la nueva escala.",
        )

    def recalculate_captured_box_sizes(self):
        try:
            px_per_cm_length, px_per_cm_width = self.get_px_per_cm()
        except ValueError:
            return

        for box in self.captured_boxes:
            box["length_cm"] = round(box["source_rotated_w"] / px_per_cm_length, 2)
            box["width_cm"] = round(box["source_rotated_h"] / px_per_cm_width, 2)

        if self.packing_started:
            self.recompute_packing()
        else:
            self.redraw_container()

    def redraw_container(self):
        try:
            container_length, container_width, container_height = self.get_container_settings()
        except ValueError:
            messagebox.showerror(
                "Contenedor invalido",
                "Largo, ancho y alto del contenedor deben ser numeros.",
            )
            return

        self.draw_container(container_length, container_width, container_height)

    def start_packing(self):
        if not self.captured_boxes:
            messagebox.showinfo(
                "Sin cajas",
                "Primero captura una o mas cajas antes de empezar el acomodo.",
            )
            return

        previous_packing_started = self.packing_started
        self.packing_started = True
        if not self.recompute_packing():
            self.packing_started = previous_packing_started
            return

        messagebox.showinfo(
            "Acomodo iniciado",
            f"Se inicio el acomodo del contenedor {self.container_number}.",
        )

    def repack_container(self):
        if not self.captured_boxes:
            messagebox.showinfo(
                "Sin cajas",
                "No hay cajas capturadas para reacomodar.",
            )
            return

        previous_packing_started = self.packing_started
        self.packing_started = True
        if not self.recompute_packing():
            self.packing_started = previous_packing_started
            return

        messagebox.showinfo(
            "Reacomodo actualizado",
            f"Se recalculo el acomodo con {len(self.captured_boxes)} caja(s).",
        )

    def recompute_packing(self):
        try:
            container_length, container_width, container_height = self.get_container_settings()
        except ValueError:
            messagebox.showerror(
                "Contenedor invalido",
                "Largo, ancho y alto del contenedor deben ser numeros.",
            )
            return False

        self.placed_boxes, self.unplaced_boxes = self.pack_boxes(
            self.captured_boxes,
            container_length,
            container_width,
        )
        self.packing_dirty = False
        self.draw_container(container_length, container_width, container_height)
        return True

    def draw_container(self, container_length, container_width, container_height):
        canvas = self.container_canvas
        canvas.delete("all")

        canvas_width = int(canvas["width"])
        canvas_height = int(canvas["height"])
        padding_x = 24
        top_padding = 34
        bottom_padding = 84

        scale = min(
            (canvas_width - padding_x * 2) / container_length,
            (canvas_height - top_padding - bottom_padding) / container_width,
        )
        draw_width = container_length * scale
        draw_height = container_width * scale
        origin_x = (canvas_width - draw_width) / 2
        origin_y = top_padding

        canvas.create_rectangle(
            origin_x,
            origin_y,
            origin_x + draw_width,
            origin_y + draw_height,
            outline="#222222",
            width=2,
            fill="#ffffff",
        )
        canvas.create_text(
            canvas_width / 2,
            12,
            text=f"Contenedor: {container_length:g} x {container_width:g} x {container_height:g} cm",
            fill="#222222",
        )

        colors = ["#8ecae6", "#ffb703", "#90be6d", "#f28482", "#bdb2ff"]

        for index, box in enumerate(self.placed_boxes):
            x1 = origin_x + box["placed_x"] * scale
            y1 = origin_y + box["placed_y"] * scale
            x2 = x1 + box["placed_length_cm"] * scale
            y2 = y1 + box["placed_width_cm"] * scale
            color = colors[index % len(colors)]

            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#111111", width=1)
            canvas.create_text(
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                text=str(box["id"]),
                fill="#111111",
            )

        if self.packing_started and self.unplaced_boxes:
            self.draw_unplaced_boxes(canvas, origin_x, origin_y + draw_height + 16, scale)

        self.update_box_table()
        self.update_container_summary()

    def draw_unplaced_boxes(self, canvas, origin_x, origin_y, scale):
        canvas.create_text(
            origin_x,
            origin_y,
            text="Sin espacio:",
            anchor="w",
            fill="#b00020",
            font=("TkDefaultFont", 9, "bold"),
        )

        current_x = origin_x + 85
        max_width = int(canvas["width"]) - 24

        for box in self.unplaced_boxes[:6]:
            width = max(28, min(box["length_cm"] * scale, 80))
            height = max(18, min(box["width_cm"] * scale, 34))

            if current_x + width > max_width:
                break

            canvas.create_rectangle(
                current_x,
                origin_y - 12,
                current_x + width,
                origin_y - 12 + height,
                fill="#ffccd5",
                outline="#b00020",
                width=1,
            )
            canvas.create_text(
                current_x + width / 2,
                origin_y - 12 + height / 2,
                text=str(box["id"]),
                fill="#b00020",
            )
            current_x += width + 6

        if len(self.unplaced_boxes) > 6:
            canvas.create_text(
                current_x,
                origin_y,
                text=f"+{len(self.unplaced_boxes) - 6}",
                anchor="w",
                fill="#b00020",
            )

        canvas.create_text(
            origin_x,
            origin_y + 34,
            text="Si no caben, calibra Px/cm largo/ancho o revisa las medidas reales.",
            anchor="w",
            fill="#b00020",
        )

    def update_box_table(self):
        if not hasattr(self, "box_tree"):
            return

        for item in self.box_tree.get_children():
            self.box_tree.delete(item)

        placed_ids = {box["id"] for box in self.placed_boxes}
        unplaced_ids = {box["id"] for box in self.unplaced_boxes}

        for box in self.captured_boxes:
            if box["id"] in placed_ids:
                status = "acomodada"
            elif box["id"] in unplaced_ids:
                status = "sin espacio"
            else:
                status = "pendiente"

            self.box_tree.insert(
                "",
                "end",
                values=(
                    box["id"],
                    f"{box['length_cm']:.2f} cm",
                    f"{box['width_cm']:.2f} cm",
                    status,
                ),
            )

    def update_container_summary(self):
        self.container_status_var.set(
            f"Contenedor {self.container_number} | Capturas: {self.capture_count} | Cajas capturadas: {len(self.captured_boxes)}"
        )

        if not self.packing_started:
            self.packing_summary_var.set("Acomodo no iniciado")
            return

        dirty_text = " | Reacomodo pendiente" if self.packing_dirty else ""
        self.packing_summary_var.set(
            f"Acomodadas: {len(self.placed_boxes)} | Sin espacio: {len(self.unplaced_boxes)}{dirty_text}"
        )

    def pack_boxes(self, boxes, container_length, container_width):
        placed = []
        unplaced = []
        current_x = 0.0
        current_y = 0.0
        row_width = 0.0

        for box in boxes:
            orientation = self.choose_orientation(
                box,
                container_length - current_x,
                container_width - current_y,
            )

            if orientation is None:
                current_y += row_width
                current_x = 0.0
                row_width = 0.0
                orientation = self.choose_orientation(
                    box,
                    container_length,
                    container_width - current_y,
            )

            if orientation is None:
                unplaced_box = dict(box)
                unplaced_box["reason"] = "No cabe en el espacio disponible"
                unplaced.append(unplaced_box)
                continue

            placed_length, placed_width, rotated = orientation
            placed_box = dict(box)
            placed_box.update(
                {
                    "placed_x": current_x,
                    "placed_y": current_y,
                    "placed_length_cm": placed_length,
                    "placed_width_cm": placed_width,
                    "rotated_for_packing": rotated,
                }
            )
            placed.append(placed_box)

            current_x += placed_length
            row_width = max(row_width, placed_width)

        return placed, unplaced

    def choose_orientation(self, box, available_length, available_width):
        length = box["length_cm"]
        width = box["width_cm"]
        options = [(length, width, False)]

        if length != width:
            options.append((width, length, True))

        valid_options = [
            option
            for option in options
            if option[0] <= available_length and option[1] <= available_width
        ]

        if not valid_options:
            return None

        return min(valid_options, key=lambda option: option[1])

    def add_detected_boxes_to_container(self, objects, timestamp, image_path):
        px_per_cm_length, px_per_cm_width = self.get_px_per_cm()
        added_boxes = []

        for obj in objects:
            length_cm = obj["rotated_w"] / px_per_cm_length
            width_cm = obj["rotated_h"] / px_per_cm_width
            box_id = len(self.captured_boxes) + 1

            box = {
                "id": box_id,
                "timestamp": timestamp,
                "image": str(image_path),
                "length_cm": round(length_cm, 2),
                "width_cm": round(width_cm, 2),
                "source_x": int(obj["x"]),
                "source_y": int(obj["y"]),
                "source_rotated_w": int(obj["rotated_w"]),
                "source_rotated_h": int(obj["rotated_h"]),
            }
            self.captured_boxes.append(box)
            added_boxes.append(box)

        return added_boxes

    def clear_packing(self):
        self.captured_boxes = []
        self.placed_boxes = []
        self.unplaced_boxes = []
        self.capture_count = 0
        self.packing_started = False
        self.packing_dirty = False
        self.redraw_container()
        self.status_var.set("Acomodo limpiado")

    def start_camera(self):
        if self.running:
            return

        try:
            camera_index, width, height = self.get_camera_settings()
        except ValueError:
            messagebox.showerror("Configuracion invalida", "Camara, ancho y alto deben ser numeros.")
            return

        self.cap = open_camera(camera_index, width, height)

        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Error", f"No se pudo abrir la camara {camera_index}.")
            return

        self.running = True
        self.last_time = time.perf_counter()
        self.status_var.set("Deteccion iniciada")
        self.update_frame()

    def stop_camera(self):
        self.running = False

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.status_var.set("Camara detenida")

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()

        if not ret:
            self.stop_camera()
            messagebox.showerror("Error", "No se pudo leer la camara.")
            return

        try:
            _, width, height = self.get_camera_settings()
            detector_args = self.build_detector_args()
        except ValueError:
            self.stop_camera()
            messagebox.showerror("Configuracion invalida", "Los filtros deben ser numeros validos.")
            return

        frame = resize_frame(frame, width, height)
        self.last_clean_frame = frame.copy()

        try:
            roi = parse_roi(self.roi_var.get().strip() or None, frame.shape)
        except ValueError as error:
            self.stop_camera()
            messagebox.showerror("ROI invalido", str(error))
            return

        detected_frame, _, objects = detect_objects(frame, detector_args, roi)
        self.last_objects = objects
        self.update_table(objects)
        self.update_fps_status(len(objects))
        self.show_frame(detected_frame)

        self.root.after(10, self.update_frame)

    def update_fps_status(self, object_count):
        current_time = time.perf_counter()
        elapsed = current_time - self.last_time
        self.last_time = current_time

        if elapsed > 0:
            self.fps = 1 / elapsed

        self.status_var.set(
            f"Modo: {self.mode_var.get()} | Objetos: {object_count} | FPS: {self.fps:.1f}"
        )

    def show_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode(".ppm", rgb_frame)

        if not success:
            return

        self.last_photo = tk.PhotoImage(data=buffer.tobytes(), format="PPM")
        self.video_label.configure(image=self.last_photo)

    def update_table(self, objects):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for index, obj in enumerate(objects, start=1):
            self.tree.insert(
                "",
                "end",
                values=(
                    index,
                    obj["x"],
                    obj["y"],
                    obj["w"],
                    obj["h"],
                    f"{obj['rotated_w']}x{obj['rotated_h']}",
                    int(obj["area"]),
                ),
            )

    def capture(self):
        if self.last_clean_frame is None:
            messagebox.showinfo("Sin captura", "Primero inicia la deteccion.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("Capturas")
        image_path = output_dir / f"captura_{timestamp}.jpg"
        json_path = output_dir / f"captura_{timestamp}.json"
        csv_path = output_dir / "medidas.csv"
        boxes_csv_path = output_dir / "cajas_contenedor.csv"
        added_boxes = []
        packing_error = None

        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            image_saved = cv2.imwrite(str(image_path), self.last_clean_frame)

            if not image_saved:
                raise OSError(f"No se pudo guardar la imagen: {image_path}")

            objects_data = self.serialize_objects(self.last_objects)

            try:
                added_boxes = self.add_detected_boxes_to_container(
                    self.last_objects,
                    timestamp,
                    image_path,
                )
            except ValueError as error:
                packing_error = str(error)

            try:
                container_length, container_width, container_height = self.get_container_settings()
                container_data = {
                    "number": self.container_number,
                    "length_cm": container_length,
                    "width_cm": container_width,
                    "height_cm": container_height,
                }
            except ValueError:
                container_data = None

            if added_boxes:
                self.capture_count += 1
                self.packing_dirty = self.packing_started

            data = {
                "timestamp": timestamp,
                "image": str(image_path),
                "mode": self.mode_var.get(),
                "roi": self.roi_var.get(),
                "px_per_cm": {
                    "length": self.px_per_cm_var.get(),
                    "width": self.px_per_cm_width_var.get(),
                },
                "reference_cm": {
                    "length": self.reference_length_var.get(),
                    "width": self.reference_width_var.get(),
                },
                "container": container_data,
                "objects": objects_data,
                "added_boxes": added_boxes,
                "placed_boxes": self.placed_boxes,
                "unplaced_boxes": self.unplaced_boxes,
                "packing_started": self.packing_started,
                "packing_dirty": self.packing_dirty,
                "packing_error": packing_error,
            }

            with json_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)

            self.append_csv(csv_path, timestamp, image_path, objects_data)
            self.append_boxes_csv(boxes_csv_path, added_boxes)
            self.redraw_container()
        except Exception as error:
            messagebox.showerror("Error al guardar captura", str(error))
            self.status_var.set("Error al guardar captura")
            return

        if packing_error:
            messagebox.showwarning(
                "Captura guardada",
                f"Se guardo la imagen, pero no se agrego al contenedor: {packing_error}",
            )
        elif not added_boxes:
            messagebox.showwarning(
                "Captura guardada",
                f"Se guardo la captura, pero no habia objetos detectados.\n\nImagen: {image_path}",
            )
        else:
            next_step = "Presiona Reacomodo de contenedor para actualizar el acomodo."

            if not self.packing_started:
                next_step = "Presiona Empezar acomodo de contenedor cuando termines de capturar."

            messagebox.showinfo(
                "Captura guardada",
                f"Imagen: {image_path}\nMedidas: {json_path}\nCajas agregadas: {len(added_boxes)}\n\n{next_step}",
            )

        self.status_var.set(f"Captura guardada: {image_path} | Medidas: {json_path}")

    def serialize_objects(self, objects):
        data = []

        for index, obj in enumerate(objects, start=1):
            data.append(
                {
                    "id": index,
                    "x": int(obj["x"]),
                    "y": int(obj["y"]),
                    "w": int(obj["w"]),
                    "h": int(obj["h"]),
                    "rotated_w": int(obj["rotated_w"]),
                    "rotated_h": int(obj["rotated_h"]),
                    "area": int(obj["area"]),
                    "aspect": float(obj["aspect"]),
                    "extent": float(obj["extent"]),
                }
            )

        return data

    def append_csv(self, csv_path, timestamp, image_path, objects):
        file_exists = csv_path.exists()

        with csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "image",
                        "object_id",
                        "x",
                        "y",
                        "w",
                        "h",
                        "rotated_w",
                        "rotated_h",
                        "area",
                        "aspect",
                        "extent",
                    ]
                )

            for obj in objects:
                writer.writerow(
                    [
                        timestamp,
                        image_path,
                        obj["id"],
                        obj["x"],
                        obj["y"],
                        obj["w"],
                        obj["h"],
                        obj["rotated_w"],
                        obj["rotated_h"],
                        obj["area"],
                        obj["aspect"],
                        obj["extent"],
                    ]
                )

    def append_boxes_csv(self, csv_path, boxes):
        if not boxes:
            return

        file_exists = csv_path.exists()

        with csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow(
                    [
                        "box_id",
                        "timestamp",
                        "image",
                        "length_cm",
                        "width_cm",
                        "source_x",
                        "source_y",
                        "source_rotated_w",
                        "source_rotated_h",
                        "px_per_cm_length",
                        "px_per_cm_width",
                    ]
                )

            px_per_cm_length, px_per_cm_width = self.get_px_per_cm()

            for box in boxes:
                writer.writerow(
                    [
                        box["id"],
                        box["timestamp"],
                        box["image"],
                        box["length_cm"],
                        box["width_cm"],
                        box["source_x"],
                        box["source_y"],
                        box["source_rotated_w"],
                        box["source_rotated_h"],
                        px_per_cm_length,
                        px_per_cm_width,
                    ]
                )

    def close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
