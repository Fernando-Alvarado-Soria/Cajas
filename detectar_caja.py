import argparse
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detecta objetos visibles con la camara usando contornos."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice de la camara a usar. Ejemplo: 0, 1, 2.",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Lista las camaras disponibles y termina el programa.",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=10,
        help="Cantidad maxima de indices de camara a revisar al listar.",
    )
    parser.add_argument(
        "--image",
        help="Ruta de una imagen para probar la deteccion sin abrir la camara.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Ancho de captura/procesamiento. Menor valor = mas velocidad.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Alto de captura/procesamiento. Menor valor = mas velocidad.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=1500,
        help="Area minima en pixeles para considerar un contorno como objeto.",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=10,
        help="Cantidad maxima de objetos a dibujar.",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Muestra la ventana de mascara/bordes. Puede bajar los FPS.",
    )
    parser.add_argument(
        "--mode",
        choices=["edge", "white"],
        default="edge",
        help="Modo de deteccion: edge detecta bordes, white detecta objetos claros.",
    )
    parser.add_argument(
        "--roi",
        help="Zona de trabajo x,y,ancho,alto. Ejemplo: --roi 80,120,500,300",
    )
    parser.add_argument(
        "--min-aspect",
        type=float,
        default=0.35,
        help="Proporcion minima ancho/alto para aceptar un objeto.",
    )
    parser.add_argument(
        "--max-aspect",
        type=float,
        default=7.0,
        help="Proporcion maxima ancho/alto para aceptar un objeto.",
    )
    parser.add_argument(
        "--min-extent",
        type=float,
        default=0.18,
        help="Que tanto llena el contorno su rectangulo. Mayor valor filtra ruido.",
    )
    parser.add_argument(
        "--save-dir",
        default="Capturas",
        help="Carpeta donde se guardan capturas limpias al presionar s.",
    )
    parser.add_argument(
        "--allow-nested",
        action="store_true",
        help="Permite mostrar objetos detectados dentro de otros objetos.",
    )
    return parser.parse_args()


def list_cameras(max_cameras):
    found = []

    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            found.append(index)
            print(f"Camara detectada en indice: {index}")
        else:
            print(f"No hay camara en indice: {index}")

        cap.release()

    if not found:
        print("No se detectaron camaras disponibles.")


def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def parse_roi(roi_text, frame_shape):
    if not roi_text:
        return None

    try:
        x, y, w, h = [int(value.strip()) for value in roi_text.split(",")]
    except ValueError:
        raise ValueError("El ROI debe tener el formato x,y,ancho,alto") from None

    frame_h, frame_w = frame_shape[:2]
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))

    return x, y, w, h


def apply_roi(mask, roi):
    if not roi:
        return mask

    roi_mask = np.zeros_like(mask)
    x, y, w, h = roi
    roi_mask[y : y + h, x : x + w] = mask[y : y + h, x : x + w]
    return roi_mask


def build_edge_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 130)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


def build_white_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 115])
    upper_white = np.array([179, 120, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def build_object_mask(frame, mode, roi):
    if mode == "white":
        mask = build_white_mask(frame)
    else:
        mask = build_edge_mask(frame)

    return apply_roi(mask, roi)


def intersection_area(first, second):
    x1 = max(first["x"], second["x"])
    y1 = max(first["y"], second["y"])
    x2 = min(first["x"] + first["w"], second["x"] + second["w"])
    y2 = min(first["y"] + first["h"], second["y"] + second["h"])

    if x2 <= x1 or y2 <= y1:
        return 0

    return (x2 - x1) * (y2 - y1)


def remove_nested_objects(objects):
    filtered = []

    for obj in objects:
        obj_bbox_area = obj["w"] * obj["h"]
        is_nested = False

        for bigger_obj in filtered:
            overlap = intersection_area(obj, bigger_obj)

            if obj_bbox_area and overlap / obj_bbox_area > 0.85:
                is_nested = True
                break

        if not is_nested:
            filtered.append(obj)

    return filtered


def find_objects(
    mask,
    min_area,
    max_objects,
    min_aspect,
    max_aspect,
    min_extent,
    allow_nested=False,
):
    frame_h, frame_w = mask.shape[:2]

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    objects = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 25 or h < 25:
            continue

        bbox_area = w * h
        frame_area = frame_w * frame_h
        aspect = w / h
        extent = area / bbox_area if bbox_area else 0

        if aspect < min_aspect or aspect > max_aspect:
            continue

        if extent < min_extent:
            continue

        if bbox_area > frame_area * 0.70:
            continue

        touches_left = x <= 2
        touches_top = y <= 2
        touches_right = x + w >= frame_w - 2
        touches_bottom = y + h >= frame_h - 2

        if (touches_left or touches_right) and (touches_top or touches_bottom):
            continue

        if (touches_left and touches_right) or (touches_top and touches_bottom):
            continue

        rotated_rect = cv2.minAreaRect(contour)
        rotated_w, rotated_h = rotated_rect[1]
        rotated_long = int(max(rotated_w, rotated_h))
        rotated_short = int(min(rotated_w, rotated_h))

        objects.append(
            {
                "contour": contour,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "aspect": aspect,
                "extent": extent,
                "rotated_rect": rotated_rect,
                "rotated_w": rotated_long,
                "rotated_h": rotated_short,
            }
        )

    objects.sort(key=lambda item: item["area"], reverse=True)

    if not allow_nested:
        objects = remove_nested_objects(objects)

    return objects[:max_objects]


def draw_objects(frame, objects):
    for index, obj in enumerate(objects, start=1):
        x = obj["x"]
        y = obj["y"]
        w = obj["w"]
        h = obj["h"]
        area = int(obj["area"])
        center_x = x + w // 2
        center_y = y + h // 2
        rotated_box = cv2.boxPoints(obj["rotated_rect"])
        rotated_box = np.intp(rotated_box)

        cv2.drawContours(frame, [rotated_box], 0, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 0), 1)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        text = f"Obj {index} | x:{x} y:{y} | rot:{obj['rotated_w']}x{obj['rotated_h']} | area:{area}"
        text_x = x
        text_y = y - 8

        if text_y < 55:
            text_y = y + h + 20

        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_w, text_h = text_size
        cv2.rectangle(
            frame,
            (text_x - 2, text_y - text_h - 4),
            (text_x + text_w + 2, text_y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def draw_status(frame, camera_index, object_count, mode, fps=None):
    if fps is None:
        text = f"Modo: {mode} | Objetos: {object_count} | Salir: q o Esc"
    else:
        text = f"Camara: {camera_index} | Modo: {mode} | Objetos: {object_count} | FPS: {fps:.1f} | s:guardar | q:salir"

    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def draw_roi(frame, roi):
    if not roi:
        return

    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(
        frame,
        "ROI",
        (x, max(y - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )


def detect_objects(frame, args, roi):
    mask = build_object_mask(frame, args.mode, roi)
    objects = find_objects(
        mask,
        args.min_area,
        args.max_objects,
        args.min_aspect,
        args.max_aspect,
        args.min_extent,
        allow_nested=args.allow_nested,
    )
    draw_roi(frame, roi)
    draw_objects(frame, objects)
    return frame, mask, objects


def save_clean_frame(frame, save_dir):
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("captura_%Y%m%d_%H%M%S.jpg")
    output_path = output_dir / filename
    cv2.imwrite(str(output_path), frame)
    return output_path


def run_image(args):
    frame = cv2.imread(args.image)

    if frame is None:
        print(f"No se pudo leer la imagen: {args.image}")
        return

    frame = resize_frame(frame, args.width, args.height)
    roi = parse_roi(args.roi, frame.shape)
    frame, mask, objects = detect_objects(frame, args, roi)
    draw_status(frame, args.camera, len(objects), args.mode)

    print(f"Objetos detectados: {len(objects)}")

    for index, obj in enumerate(objects, start=1):
        print(
            f"Obj {index}: x={obj['x']} y={obj['y']} "
            f"ancho={obj['w']} alto={obj['h']} area={int(obj['area'])} "
            f"aspect={obj['aspect']:.2f} extent={obj['extent']:.2f}"
        )

    cv2.imshow("Deteccion de objetos", frame)

    if args.show_mask:
        cv2.imshow("Mascara", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_camera(camera_index, width, height):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def run_camera(args):
    cap = open_camera(args.camera, args.width, args.height)

    if not cap.isOpened():
        print(f"No se pudo abrir la camara {args.camera}")
        print("Prueba listar camaras con: python detectar_caja.py --list-cameras")
        return

    last_time = time.perf_counter()
    fps = 0.0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer la camara")
            break

        frame = resize_frame(frame, args.width, args.height)
        clean_frame = frame.copy()
        roi = parse_roi(args.roi, frame.shape)
        frame, mask, objects = detect_objects(frame, args, roi)

        current_time = time.perf_counter()
        elapsed = current_time - last_time
        last_time = current_time

        if elapsed > 0:
            fps = 1 / elapsed

        draw_status(frame, args.camera, len(objects), args.mode, fps)

        cv2.imshow("Deteccion de objetos", frame)

        if args.show_mask:
            cv2.imshow("Mascara", mask)

        key = cv2.waitKey(1)

        if key == ord("s"):
            output_path = save_clean_frame(clean_frame, args.save_dir)
            print(f"Captura guardada: {output_path}")

        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    if args.list_cameras:
        list_cameras(args.max_cameras)
        return

    if args.image:
        run_image(args)
        return

    run_camera(args)


if __name__ == "__main__":
    main()
