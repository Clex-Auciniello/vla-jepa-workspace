"""Preparazione online dei video raccolti dal client UR5e.

Input:
    Archivio ZIP contenente manifest.json e coppie PNG RGB
    delle camere front e gripper.

Output:
    Un batch PyTorch per finestra:

    {
        '<front_feature>':   float32 [1, 8, 3, 224, 224],
        '<gripper_feature>': float32 [1, 8, 3, 224, 224],
        'task': [prompt_di_movimento],
    }

    I tensori RGB hanno valori in [0, 1].

    Finestre da 8 frame, stride 5, batch size 1.
    Nessuna normalizzazione ImageNet aggiuntiva.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from PIL import Image
import cv2


@dataclass(frozen=True)
class CropMargins:
    """Margini (pixel originali) da tagliare PRIMA del resize.

    Non confondere con le coordinate di un bounding box [x1,y1,x2,y2].
    Se un crop è richiesto dalla baseline, convertirlo esplicitamente in margini.
    """

    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0


@dataclass(frozen=True)
class OnlineBatchBuilderConfig:
    # L'obiettivo WM della revisione LeRobot usa 8 frame e due viste.
    window_size: int = 8
    stride: int = 5
    batch_size: int = 1
    image_size: tuple[int, int] = (224, 224)  # (width, height)

    # Crop FRONTALE IDENTICO a quello richiesto per il futuro fine-tuning
    # e usato per il campionamento degli otto vertici: margini pixel originali
    #  PRIMA del resize a 224x224.
    # La camera gripper non e' croppata: nessuna geometria non verificata.
    # La baseline usa (top, bottom, left, right) = (0, 10, 140, 90).
    # CropMargins richiede invece (left, top, right, bottom).
    front_crop: CropMargins | None = CropMargins(
        left=140,
        top=0,
        right=90,
        bottom=10,
    )
    gripper_crop: CropMargins | None = None

    # I nomi derivano dal checkpoint DROID/UR5e mostrato dall'utente.
    # Vengono comunque confrontati con le feature della policy dal trainer.
    front_feature: str = "observation.images.exterior_1_left"
    gripper_feature: str = "observation.images.exterior_2_left"

    # Limiti difensivi per dati caricati da un client remoto.
    max_frames: int = 2000


@dataclass(frozen=True)
class EpisodeWindow:
    """Una finestra pronta per il trainer, più metadati per log/debug.

    `start_index` è la posizione 0-based nella sequenza dei frame registrati.
    """

    episode_id: str
    window_index: int
    start_index: int
    batch: dict[str, Any]


def _read_manifest(zf: zipfile.ZipFile, cfg: OnlineBatchBuilderConfig) -> dict[str, Any]:
    """Valida il manifest e i path prima di decodificare i PNG."""
    names = zf.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Archivio con nomi di file ZIP duplicati")
    if "manifest.json" not in names:
        raise ValueError("Archivio senza manifest.json")

    manifest = json.loads(zf.read("manifest.json"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Manifest non conforme allo schema versione 1")
    if manifest.get("sampling_mode") != "motion_step":
        raise ValueError("Il batch builder richiede sampling_mode='motion_step'")
    episode_id = manifest.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise ValueError("episode_id mancante o non valido")

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not (cfg.window_size <= len(frames) <= cfg.max_frames):
        raise ValueError(
            f"Numero di frame non valido: servono almeno {cfg.window_size} "
            f"e massimo {cfg.max_frames}"
        )

    # Verifica l'ordine definito dal recorder (indici da 1 a N) e che le coppie
    # front/gripper non puntino ad altri file o a percorsi arbitrari nello ZIP.
    for expected_index, frame in enumerate(frames, start=1):
        if not isinstance(frame, dict) or frame.get("index") != expected_index:
            raise ValueError(f"Frame non ordinato o indice errato alla posizione {expected_index}")
        for camera, field in (
            ("front", "front_path"),
            ("gripper", "gripper_path"),
        ):
            path = f"frames/{camera}/{expected_index:06d}.png"
            if frame.get(field) != path or path not in names:
                raise ValueError(f"PNG mancante o path inatteso: {path}")

        label = frame.get("motion_label")
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ValueError(f"motion_label non valido per frame {expected_index}")
    plays = _is_plays_manifest(manifest)
    if not plays:
        # Percorso Exploration IDENTICO al precedente: controlla ciascuna
        # finestra completa con la stessa funzione di prompt originale.
        for start in range(0, len(frames) - cfg.window_size + 1, cfg.stride):
            _prompt_for_window(frames[start : start + cfg.window_size])
    return manifest


def _is_plays_manifest(manifest: Mapping[str, Any]) -> bool:
    """I metadati selezionano SOLO il preprocessing, non vengono dati al trainer.

    Exploration non include 'source'/'prompt_mode': resta retrocompatibile.
    Plays usa prompt unico e finestra finale allineata (senza padding reale).
    """
    if manifest.get("source") == "plays":
        if manifest.get("prompt_mode") != "episode_prompt":
            raise ValueError("Plays richiede prompt_mode='episode_prompt'")
        if manifest.get("terminal_padding") != "repeat_last":
            raise ValueError("Plays richiede terminal_padding='repeat_last'")
        prompt = manifest.get("episode_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Plays senza episode_prompt valido")
        return True
    if (manifest.get("prompt_mode") == "episode_prompt" or
            manifest.get("terminal_padding") == "repeat_last"):
        raise ValueError("Manifest con campi Plays, ma source diverso da 'plays'")
    return False


def _window_starts(num_frames: int, *, window_size: int, stride: int,
                   repeat_last: bool) -> list[int]:
    """Inizi finestre; Exploration = solo finestre complete, senza modifiche.

    Per Plays riallinea SOLO l'ultima finestra di 8 frame REALI all'ultimo
    frame dell'episodio (nessuna duplicazione e nessuna finestra aggiuntiva).
    Gli inizi precedenti conservano stride nominale 5.
    `repeat_last` e' mantenuto come parametro storico di compatibilita'
    con il manifest Plays terminal_padding='repeat_last'.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size e stride devono essere positivi")
    if num_frames < window_size:
        return []
    starts = list(range(0, num_frames - window_size + 1, stride))
    if repeat_last and starts[-1] + window_size < num_frames:
        # Modifica soltanto l'ultimo start; mantiene il numero di finestre.
        # Per 15 frame: start 0,5 -> 0,7 (finestre 1-8 e 8-15).
        # Con 9-12 frame c'e' una sola finestra, ancorata alla fine.
        starts[-1] = num_frames - window_size
    return starts


def _validate_all_pngs(zf: zipfile.ZipFile, manifest: Mapping[str, Any],
                       cfg: OnlineBatchBuilderConfig) -> None:
    """Preflight di TUTTI i PNG, prima di qualsiasi forward/backward.

    Verifica formato PNG, CRC/struttura con verify(), pixel effettivamente
    decodificabili con load(), crop applicabile e dimensioni coerenti per vista.
    Non effettua resize, normalizzazione o preparazione dei tensori.
    """
    sizes: dict[str, tuple[int, int]] = {}
    for frame in manifest["frames"]:
        for camera, field, crop in (
            ("front", "front_path", cfg.front_crop),
            ("gripper", "gripper_path", cfg.gripper_crop),
        ):
            path = frame[field]
            try:
                data = zf.read(path)
                with Image.open(io.BytesIO(data)) as img:
                    if img.format != "PNG":
                        raise ValueError("Formato non PNG")
                    size = img.size
                    if size[0] <= 0 or size[1] <= 0:
                        raise ValueError("Dimensioni PNG non valide")
                    img.verify()  # verifica chunk e CRC (incl. dati IDAT)
                with Image.open(io.BytesIO(data)) as img:
                    img.load()    # verifica che i pixel siano decodificabili
                    img.convert("RGB").load()
                if crop is not None:
                    if (crop.left < 0 or crop.top < 0 or crop.right < 0 or crop.bottom < 0 or
                            size[0] - crop.left - crop.right <= 0 or
                            size[1] - crop.top - crop.bottom <= 0):
                        raise ValueError(f"Crop non applicabile a immagine {size}: {crop}")
                if camera in sizes and sizes[camera] != size:
                    raise ValueError(f"Dimensione camera {camera} variabile: {sizes[camera]} -> {size}")
                sizes[camera] = size
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(f"PNG {path} non valido: {exc}") from exc


def _prompt_for_window(frames: list[Mapping[str, Any]]) -> str:
    """Crea il prompt dalle SETTE transizioni verso i frame 2..8.

    Il primo frame fornisce l'osservazione iniziale e non la transizione entrante.
    Rimuoviamo solo ripetizioni ADIACENTI: un movimento che ricompare più tardi
    nella finestra deve restare nel prompt.
    """
    motions: list[str] = []
    for frame in frames[1:]:
        raw_label = frame.get("motion_label")
        if raw_label is None:
            raise ValueError("Transizione senza motion_label dentro una finestra")
        label = " ".join(raw_label.strip().rstrip(".").split())
        if label and (not motions or motions[-1] != label):
            motions.append(label)
    if not motions:
        raise ValueError("Finestra senza movimenti utilizzabili")
    # Il testo successivo a 'then' continua la stessa frase: 'move', 'close'...
    return motions[0] + "".join(", then " + m[0].lower() + m[1:] for m in motions[1:]) + "."


def _image_to_tensor(
    zf: zipfile.ZipFile,
    path: str,
    crop: CropMargins | None,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Da PNG RGB a float32 [3,224,224] in [0,1], come nella baseline.

    Il processor V-JEPA gestisce le trasformazioni successive.
    """
    with Image.open(io.BytesIO(zf.read(path))) as source:
        rgb = source.convert("RGB")
        if crop is not None:
            x0, y0 = crop.left, crop.top
            x1, y1 = rgb.width - crop.right, rgb.height - crop.bottom
            if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
                raise ValueError(f"Crop impossibile per {path}: dimensione {rgb.size}, crop {crop}")
            rgb = rgb.crop((x0, y0, x1, y1))
        array = np.asarray(rgb, dtype=np.uint8)
        array = cv2.resize(
            array,
            image_size,
            interpolation=cv2.INTER_LINEAR,
        )
    return (
        torch.from_numpy(np.ascontiguousarray(array))
        .permute(2, 0, 1)
        .contiguous()
        .to(torch.float32)
        .div_(255.0)
    )


class OnlineBatchBuilder:
    """Converte in streaming ogni episodio ZIP nelle finestre usate da VLA-JEPA.

    Non effettua l'addestramento e non incrementa contatori: il trainer decide
    quando un episodio è completo e quando salvare i checkpoint.
    """

    def __init__(self, cfg: OnlineBatchBuilderConfig | None = None) -> None:
        self.cfg = cfg or OnlineBatchBuilderConfig()
        if self.cfg.window_size != 8 or self.cfg.batch_size != 1:
            raise ValueError("Questa versione VLA-JEPA usa finestre da 8 e batch fisico da 1")
        if self.cfg.stride <= 0 or any(d <= 0 for d in self.cfg.image_size):
            raise ValueError("stride e image_size devono essere positivi")
        if self.cfg.front_feature == self.cfg.gripper_feature:
            raise ValueError("Le feature front e gripper devono essere distinte")

    def inspect_episode(self, archive: bytes | str | Path) -> dict[str, Any]:
        """Valida manifest e TUTTI i PNG prima della messa in coda GPU."""
        source = io.BytesIO(archive) if isinstance(archive, bytes) else archive
        with zipfile.ZipFile(source, "r") as zf:
            manifest = _read_manifest(zf, self.cfg)
            _validate_all_pngs(zf, manifest, self.cfg)
            return {
                "episode_id": manifest["episode_id"],
                "frames": len(manifest["frames"]),
                "windows": self.count_windows(
                    len(manifest["frames"]),
                    window_size=self.cfg.window_size,
                    stride=self.cfg.stride,
                    repeat_last=_is_plays_manifest(manifest),
                ),
            }

    def iter_episode_windows(self, archive: bytes | str | Path) -> Iterator[EpisodeWindow]:
        """Yield lazy: soltanto una finestra RGB è in RAM per volta.

        La sorgente `archive` è lo ZIP ricevuto dal server come bytes oppure
        un file .zip locale. Tenere in vita il generatore mentre si processano
        i batch, altrimenti il file ZIP si chiude alla fine dell'iterazione.
        """
        source = io.BytesIO(archive) if isinstance(archive, bytes) else archive
        with zipfile.ZipFile(source, "r") as zf:
            manifest = _read_manifest(zf, self.cfg)
            # Anche nell'utilizzo standalone dell'iteratore controlliamo i PNG
            # prima di produrre una finestra: nessun training parziale per
            # corruzioni dei file che sono gia' presenti nello ZIP.
            _validate_all_pngs(zf, manifest, self.cfg)
            frames: list[dict[str, Any]] = manifest["frames"]
            episode_id: str = manifest["episode_id"]
            plays = _is_plays_manifest(manifest)

            for window_i, start in enumerate(_window_starts(
                len(frames), window_size=self.cfg.window_size,
                stride=self.cfg.stride, repeat_last=plays,
            )):
                selected = frames[start : start + self.cfg.window_size]
                task = manifest["episode_prompt"] if plays else _prompt_for_window(selected)
                front = torch.stack([
                    _image_to_tensor(zf, f["front_path"], self.cfg.front_crop, self.cfg.image_size)
                    for f in selected
                ])
                gripper = torch.stack([
                    _image_to_tensor(zf, f["gripper_path"], self.cfg.gripper_crop, self.cfg.image_size)
                    for f in selected
                ])
                # `unsqueeze(0)` aggiunge la dimensione di batch B=1.
                yield EpisodeWindow(
                    episode_id=episode_id,
                    window_index=window_i,
                    start_index=start,
                    batch={
                        self.cfg.front_feature: front.unsqueeze(0),
                        self.cfg.gripper_feature: gripper.unsqueeze(0),
                        "task": [task],
                    },
                )

    @staticmethod
    def count_windows(num_frames: int, *, window_size: int = 8, stride: int = 5,
                      repeat_last: bool = False) -> int:
        """Default Exploration invariato; repeat_last=True allinea ultimo Plays."""
        return len(_window_starts(
            num_frames, window_size=window_size,
            stride=stride, repeat_last=repeat_last,
        ))
