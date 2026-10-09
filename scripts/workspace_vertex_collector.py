#!/usr/bin/env python3
"""Acquisisce HOME e gli 8 vertici del workspace UR5e con anteprima front RGB.

Eseguire nel container ROS 2 con MoveIt e camera frontale attivi.
All'avvio (salvo --preview-only o --resume) esegue GoHome -> GoToPose.
Dopo il movimento, premere INVIO nella finestra OpenCV per salvare HOME,
poi spostare manualmente il robot e premere INVIO per ciascuno degli 8 vertici.
Non comanda altri movimenti dopo il posizionamento iniziale.
"""

import argparse
import csv
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from moveit_controller_srvs.srv import GoHome, GoToPose
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener


FRONT_TOPIC = "/zed_front/zed_node/rgb/color/rect/image"
BASE_FRAME = "base_link"
TCP_FRAME = "tcp_link"
HOME_SERVICE = "set_robot_to_home"
POSE_SERVICE = "set_robot_to_pose"

# Uguali a process_front_image() in vla_jepa_utils.py.
# Margini: (top, bottom, left, right), in pixel dell'immagine originale.
FRONT_CROP_MARGINS = (0, 10, 140, 90)
IMAGE_SIZE = 224

# Uguale a pose_before_first_inference di ai_controller_node.
# [x, y, z, qx, qy, qz, qw] nel frame base_link.
INITIAL_POSE = (
    -0.15552094619366708,
    0.34869994018501943,
    0.1532803451753288,
    0.9994452044624775,
    0.03161651380119412,
    0.0021438049655468088,
    0.010251021036213035,
)

# Ordine di acquisizione visto dalla camera frontale.
# X locale: sinistra -> destra; Y locale: dietro -> avanti; Z: basso -> alto.
VERTICES = (
    ("V000", "BASSO - dietro a sinistra"),
    ("V100", "BASSO - dietro a destra"),
    ("V010", "BASSO - avanti a sinistra"),
    ("V110", "BASSO - avanti a destra"),
    ("V001", "ALTO - dietro a sinistra"),
    ("V101", "ALTO - dietro a destra"),
    ("V011", "ALTO - avanti a sinistra"),
    ("V111", "ALTO - avanti a destra"),
)

CSV_FIELDS = (
    "vertex_id", "stamp_sec", "stamp_nanosec",
    "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw",
)


class WorkspaceCollector(Node):
    def __init__(self):
        super().__init__("workspace_vertex_collector")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

        self.bridge = CvBridge()
        self._image_lock = threading.Lock()
        self._image = None
        self._image_received_at = 0.0
        self.create_subscription(
            Image, FRONT_TOPIC, self._image_callback, qos_profile_sensor_data
        )

        self.home_client = self.create_client(GoHome, HOME_SERVICE)
        self.pose_client = self.create_client(GoToPose, POSE_SERVICE)

    def _image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._image_lock:
            self._image = frame
            self._image_received_at = time.monotonic()

    def current_image(self):
        with self._image_lock:
            if self._image is None:
                return None, False
            return self._image.copy(), time.monotonic() - self._image_received_at < 2.0

    def current_tcp_pose(self):
        tf = self.tf_buffer.lookup_transform(BASE_FRAME, TCP_FRAME, Time())
        stamp = tf.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        if stamp_ns:
            age_s = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
            if abs(age_s) > 2.0:
                raise RuntimeError(f"TF {TCP_FRAME} non aggiornato ({age_s:.1f} s)")
        p = tf.transform.translation
        q = tf.transform.rotation
        return {
            "stamp_sec": stamp.sec,
            "stamp_nanosec": stamp.nanosec,
            "x_m": float(p.x),
            "y_m": float(p.y),
            "z_m": float(p.z),
            "qx": float(q.x),
            "qy": float(q.y),
            "qz": float(q.z),
            "qw": float(q.w),
        }

    @staticmethod
    def _call_moveit(client, request):
        # L'executor ROS gira in un thread separato.
        future = client.call_async(request)
        deadline = time.monotonic() + 120.0
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("Timeout del servizio MoveIt: verificare lo stato del robot")
            time.sleep(0.05)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f"Movimento MoveIt fallito: {getattr(response, 'message', '')}")

    def move_to_home(self):
        for service_name, client in (
            (HOME_SERVICE, self.home_client),
            (POSE_SERVICE, self.pose_client),
        ):
            if not client.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f"Servizio MoveIt non disponibile: {service_name}")

        print("[MoveIt] GoHome...")
        self._call_moveit(self.home_client, GoHome.Request())

        print("[MoveIt] Posizionamento iniziale...")
        req = GoToPose.Request()
        req.pose.header.stamp = self.get_clock().now().to_msg()
        req.pose.header.frame_id = BASE_FRAME
        p = req.pose.pose.position
        q = req.pose.pose.orientation
        p.x, p.y, p.z = INITIAL_POSE[:3]
        q.x, q.y, q.z, q.w = INITIAL_POSE[3:]
        self._call_moveit(self.pose_client, req)
        print("[MoveIt] Posizione iniziale raggiunta. HOME non ancora salvata.")


def preprocess_front_preview(frame):
    """Stesso crop e resize del controller VLA-JEPA (solo parte visiva)."""
    height, width = frame.shape[:2]
    top, bottom, left, right = FRONT_CROP_MARGINS
    if top + bottom >= height or left + right >= width:
        raise ValueError(f"Crop {FRONT_CROP_MARGINS} non valido per {width}x{height}")
    cropped = frame[top:height - bottom, left:width - right]
    return cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def show_images(frame, next_point):
    """Due viste affiancate, senza disegni sull'immagine originale."""
    preview_height = 430
    raw_width = round(frame.shape[1] * preview_height / frame.shape[0])
    original = cv2.resize(frame, (raw_width, preview_height), interpolation=cv2.INTER_LINEAR)

    processed_224 = preprocess_front_preview(frame)
    processed = cv2.resize(
        processed_224, (preview_height, preview_height), interpolation=cv2.INTER_LINEAR
    )  # Solo ingrandimento per la finestra: l'input VLA-JEPA rimane 224x224.

    margin, top = 16, 80
    canvas = np.full(
        (top + preview_height + 18, raw_width + preview_height + 3 * margin, 3),
        30, dtype=np.uint8,
    )
    x_processed = 2 * margin + raw_width
    canvas[top:top + preview_height, margin:margin + raw_width] = original
    canvas[top:top + preview_height, x_processed:x_processed + preview_height] = processed

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "FRONT ORIGINALE", (margin, 29), font, 0.60,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "VLA-JEPA: CROP + 224x224", (x_processed, 29), font, 0.60,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{next_point} | INVIO: salva | U: annulla | Q: esci",
                (margin, 58), font, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


def save(path, home, vertices):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "base_frame": BASE_FRAME,
        "tcp_frame": TCP_FRAME,
        "front_camera_topic": FRONT_TOPIC,
        "front_crop_margins": list(FRONT_CROP_MARGINS),
        "home": home,
        "vertices": vertices,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        if home is not None:
            writer.writerow(home)
        writer.writerows(vertices)


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("base_frame") != BASE_FRAME or data.get("tcp_frame") != TCP_FRAME:
        raise ValueError("I frame del file salvato non coincidono con quelli configurati")
    vertices = data.get("vertices", [])
    expected = [name for name, _ in VERTICES[:len(vertices)]]
    if [v["vertex_id"] for v in vertices] != expected:
        raise ValueError("Ordine dei vertici nel file non valido")
    return data.get("home"), vertices


def next_capture(home, vertices):
    if home is None:
        return "HOME", "HOME - posa iniziale MoveIt"
    if len(vertices) < len(VERTICES):
        return VERTICES[len(vertices)]
    return None, None


def collect(node, output, home, vertices, preview_only):
    window = "UR5e - Workspace"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("[FINESTRA] Seleziona la finestra OpenCV, poi premi INVIO per salvare.")
    print("[FINESTRA] Premi U per annullare l'ultima acquisizione; Q o ESC per uscire.")
    previous_step = None
    try:
        while rclpy.ok():
            point_id, label = next_capture(home, vertices)
            if point_id is None and not preview_only:
                label = "COMPLETATO - INVIO: termina, U: annulla"
            if label != previous_step and not preview_only:
                if point_id is None:
                    print("\n[COMPLETATO] HOME e 8 vertici salvati. INVIO: termina; U: correggi V111.")
                else:
                    print(f"\n[PROSSIMO] {label} -> sposta il TCP e premi INVIO")
                previous_step = label

            frame, fresh = node.current_image()
            if frame is not None:
                description = label if not preview_only else "SOLA ANTEPRIMA"
                cv2.imshow(window, show_images(frame, description))

            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), 27):
                break
            if key == ord("u") and not preview_only:
                if vertices:
                    removed = vertices.pop()["vertex_id"]
                elif home is not None:
                    removed = "HOME"
                    home = None
                else:
                    print("[ANNULLA] Nessuna acquisizione da cancellare.")
                    continue
                save(output, home, vertices)
                previous_step = None
                print(f"[ANNULLATO] {removed}. Riposiziona il TCP e premi INVIO.")
                continue
            if key not in (10, 13) or preview_only:
                continue
            if point_id is None:
                break
            if not fresh:
                print("[ATTESA] Camera frontale non aggiornata: punto non salvato")
                continue
            try:
                pose = node.current_tcp_pose()
            except (TransformException, RuntimeError) as exc:
                print(f"[TF] Impossibile acquisire il TCP: {exc}")
                continue

            row = {"vertex_id": point_id, **pose}
            if point_id == "HOME":
                home = row
            else:
                vertices.append(row)
            save(output, home, vertices)
            print(f"[SALVATO] {point_id}: "
                  f"x={pose['x_m']:.5f}, y={pose['y_m']:.5f}, z={pose['z_m']:.5f} m")
    finally:
        cv2.destroyAllWindows()

    if not preview_only:
        print(f"\nRisultato: HOME={'OK' if home else 'mancante'}, "
              f"vertici={len(vertices)}/8")
        print(f"JSON: {output.resolve()}")
        print(f"CSV:  {output.with_suffix('.csv').resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("configs/workspace_vertices.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Riprendi un'acquisizione interrotta")
    mode.add_argument("--overwrite", action="store_true", help="Ricomincia da zero")
    parser.add_argument("--preview-only", action="store_true",
                        help="Mostra le immagini senza muovere o salvare")
    args = parser.parse_args()

    if args.output.exists() and not (args.resume or args.overwrite or args.preview_only):
        parser.error("Il file esiste gia': usa --resume oppure --overwrite")
    home, vertices = (load(args.output) if args.resume and args.output.exists()
                      else (None, []))

    rclpy.init()
    node = WorkspaceCollector()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        # Prima di comandare MoveIt, verifichiamo che la camera sia presente.
        deadline = time.monotonic() + 10.0
        while not node.current_image()[1] and time.monotonic() < deadline:
            time.sleep(0.1)
        frame, fresh = node.current_image()
        if not fresh:
            raise RuntimeError(f"Camera frontale non disponibile: {FRONT_TOPIC}")
        preprocess_front_preview(frame)

        if not args.preview_only and home is None:
            print("\n[ATTENZIONE] Movimento automatico con MoveIt: GoHome -> GoToPose.")
            print("Verifica prima che il robot non sia in Freedrive e che il percorso sia libero.")
            node.move_to_home()
        elif home is not None:
            print("[RESUME] HOME caricata; non viene comandato alcun movimento iniziale.")

        collect(node, args.output, home, vertices, args.preview_only)
    except KeyboardInterrupt:
        print("\nInterrotto da tastiera")
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
