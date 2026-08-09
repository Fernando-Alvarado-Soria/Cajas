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

        self.add_input(advanced, "Max objetos", self.max_objects_var, 0)
        self.add_input(advanced, "Aspect min", self.min_aspect_var, 1)
        self.add_input(advanced, "Aspect max", self.max_aspect_var, 2)
        self.add_input(advanced, "Extent min", self.min_extent_var, 3)

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

        status = ttk.Label(main, textvariable=self.status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))

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

        output_dir = Path("Capturas")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = output_dir / f"captura_{timestamp}.jpg"
        json_path = output_dir / f"captura_{timestamp}.json"
        csv_path = output_dir / "medidas.csv"

        cv2.imwrite(str(image_path), self.last_clean_frame)

        objects_data = self.serialize_objects(self.last_objects)
        data = {
            "timestamp": timestamp,
            "image": str(image_path),
            "mode": self.mode_var.get(),
            "roi": self.roi_var.get(),
            "objects": objects_data,
        }

        with json_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        self.append_csv(csv_path, timestamp, image_path, objects_data)
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

    def close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
