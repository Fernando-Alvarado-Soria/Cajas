import argparse
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


def build_object_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 130)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask


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


def find_objects(mask, min_area, max_objects, allow_nested=False):
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

        objects.append(
            {
                "contour": contour,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
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

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        text = f"Obj {index}: x:{x} y:{y} w:{w} h:{h} area:{area}"
        cv2.putText(
            frame,
            text,
            (x, max(y - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )


def draw_status(frame, camera_index, object_count, fps=None):
    if fps is None:
        text = f"Objetos: {object_count} | Salir: q o Esc"
    else:
        text = f"Camara: {camera_index} | Objetos: {object_count} | FPS: {fps:.1f} | Salir: q o Esc"

    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )


def detect_objects(frame, min_area, max_objects, allow_nested=False):
    mask = build_object_mask(frame)
    objects = find_objects(mask, min_area, max_objects, allow_nested=allow_nested)
    draw_objects(frame, objects)
    return frame, mask, objects


def run_image(args):
    frame = cv2.imread(args.image)

    if frame is None:
        print(f"No se pudo leer la imagen: {args.image}")
        return

    frame = resize_frame(frame, args.width, args.height)
    frame, mask, objects = detect_objects(
        frame,
        args.min_area,
        args.max_objects,
        allow_nested=args.allow_nested,
    )
    draw_status(frame, args.camera, len(objects))

    print(f"Objetos detectados: {len(objects)}")

    for index, obj in enumerate(objects, start=1):
        print(
            f"Obj {index}: x={obj['x']} y={obj['y']} "
            f"ancho={obj['w']} alto={obj['h']} area={int(obj['area'])}"
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
        frame, mask, objects = detect_objects(
            frame,
            args.min_area,
            args.max_objects,
            allow_nested=args.allow_nested,
        )

        current_time = time.perf_counter()
        elapsed = current_time - last_time
        last_time = current_time

        if elapsed > 0:
            fps = 1 / elapsed

        draw_status(frame, args.camera, len(objects), fps)

        cv2.imshow("Deteccion de objetos", frame)

        if args.show_mask:
            cv2.imshow("Mascara", mask)

        key = cv2.waitKey(1)

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
