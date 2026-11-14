
#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import savers  # Necessario per deserializzare savers.Trajectory


DEFAULT_FRONT_KEY = "camera_front_image"
DEFAULT_GRIPPER_KEY = "camera_gripper_image"


def get_step(traj, i):
    if hasattr(traj, "get"):
        try:
            return traj.get(i, decompress=False)
        except TypeError:
            return traj.get(i)
    return traj[i]


def save_image_unchanged(image, output_path):
    """
    Salva i valori RGB esattamente come sono nel PKL.

    Nessuna conversione di colore, crop, resize
    o normalizzazione.
    """

    arr = np.asarray(image)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Formato immagine non supportato: {arr.shape}. "
            "Atteso array HxWx3 già decodificato."
        )

    if arr.dtype != np.uint8:
        raise ValueError(
            f"Tipo immagine non supportato: {arr.dtype}. "
            "Atteso uint8. Nessuna conversione automatica."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Interpreta direttamente i tre canali come RGB.
    # Non modifica alcun valore dell'array.
    Image.fromarray(arr).save(
        output_path,
        format="PNG"
    )

    # Verifica che il PNG conservi esattamente i pixel.
    with Image.open(output_path) as png:
        saved = np.asarray(png)

    if not np.array_equal(arr, saved):
        raise RuntimeError(
            f"Pixel modificati durante il salvataggio: "
            f"{output_path}"
        )

    return arr.shape


def export_trajectory(
    pkl_path,
    output_dir,
    front_key,
    gripper_key,
    max_frames=None,
):
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)

    traj = (
        payload["traj"]
        if isinstance(payload, dict) and "traj" in payload
        else payload
    )

    total = len(traj)

    if max_frames is None:
        n_frames = total
    else:
        n_frames = min(total, max_frames)

    cameras = {
        "front": front_key,
        "gripper": gripper_key,
    }

    print(f"\nPKL: {pkl_path}")
    print(f"Frame totali: {total}")
    print(f"Frame da esportare: {n_frames}")

    for i in range(n_frames):
        step = get_step(traj, i)
        obs = step["obs"]

        for camera_name, key in cameras.items():

            if key not in obs:
                raise KeyError(
                    f"Frame {i}: chiave assente '{key}'"
                )

            output_path = (
                output_dir
                / camera_name
                / f"frame_{i:03d}.png"
            )

            shape = save_image_unchanged(
                obs[key],
                output_path
            )

            if i == 0:
                print(
                    f"  {camera_name}: "
                    f"{shape}, RGB uint8"
                )

    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Esporta i frame RGB da PKL preprocessati "
            "senza applicare trasformazioni."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Singolo PKL oppure cartella dataset"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Cartella di destinazione dei PNG"
    )

    parser.add_argument(
        "--front-key",
        default=DEFAULT_FRONT_KEY
    )

    parser.add_argument(
        "--gripper-key",
        default=DEFAULT_GRIPPER_KEY
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Esporta solo i primi N frame di ogni PKL"
    )

    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()

    if not source.exists():
        raise FileNotFoundError(source)

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames deve essere positivo")

    if source.is_file():
        if source.suffix != ".pkl":
            parser.error("--input deve essere un PKL")

        paths = [source]

    else:
        paths = sorted(source.rglob("*.pkl"))

        if not paths:
            raise RuntimeError(
                f"Nessun PKL trovato in {source}"
            )

    print(f"Traiettorie trovate: {len(paths)}")

    for path in paths:

        if source.is_file():
            output_dir = destination

        else:
            relative = path.relative_to(source)
            output_dir = (
                destination / relative.with_suffix("")
            )

        export_trajectory(
            pkl_path=path,
            output_dir=output_dir,
            front_key=args.front_key,
            gripper_key=args.gripper_key,
            max_frames=args.max_frames,
        )

    print("\nESPORTAZIONE COMPLETATA")
    print("Nessuna trasformazione applicata alle immagini.")


if __name__ == "__main__":
    main()
