"""Preparazione online dei video raccolti dal client UR5e.

Input: archivio ZIP generato da FlaskTrainingTransport.encode(), contenente
`manifest.json` e una coppia JPEG front/gripper per ogni step acquisito.

Output: un batch PyTorch *per finestra*, senza spostarlo sulla GPU:
    {
        '<front_feature>':   uint8 [1, 8, 3, 224, 224],
        '<gripper_feature>': uint8 [1, 8, 3, 224, 224],
        'task':              [prompt_di_movimento],
    }

Il generatore decodifica soltanto gli 8+8 JPEG necessari alla finestra corrente;
non memorizza l'episodio intero come tensore RGB. Non effettua augmentations:
queste ultime, se richieste, dovranno essere introdotte esplicitamente.
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
    # (left=0, top=10, right=140, bottom=90), PRIMA del resize a 224x224.
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
    max_jpeg_bytes: int = 12 * 1024 * 1024
    max_manifest_bytes: int = 2 * 1024 * 1024


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
    """Valida la struttura e i path prima di leggere i file JPEG."""
    names = zf.namelist()
    if len(names) != len(set(names)):
        raise ValueError("Archivio con nomi di file ZIP duplicati")
    if "manifest.json" not in names:
        raise ValueError("Archivio senza manifest.json")
    if zf.getinfo("manifest.json").file_size > cfg.max_manifest_bytes:
        raise ValueError("manifest.json troppo grande")

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
        for camera, field in (("front", "front_path"), ("gripper", "gripper_path")):
            path = f"frames/{camera}/{expected_index:06d}.jpg"
            if frame.get(field) != path or path not in names:
                raise ValueError(f"JPEG mancante o path inatteso: {path}")
            if zf.getinfo(path).file_size > cfg.max_jpeg_bytes:
                raise ValueError(f"JPEG troppo grande: {path}")
        label = frame.get("motion_label")
        if label is not None and (not isinstance(label, str) or not label.strip()):
            raise ValueError(f"motion_label non valido per frame {expected_index}")
    # Valida in anticipo le etichette di TUTTE le finestre selezionate.
    # Cosi' un prompt mancante in fondo all'episodio viene rilevato prima di
    # iniziare il training (evita aggiornamenti parziali per questa causa).
    for start in range(0, len(frames) - cfg.window_size + 1, cfg.stride):
        _prompt_for_window(frames[start : start + cfg.window_size])
    return manifest


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
    """Da JPEG RGB a uint8 [3,224,224]; nessuna normalizzazione ImageNet.

    `VLAJEPAPolicy._prepare_model_inputs()` converte già uint8 -> PIL e il
    processor V-JEPA esegue separatamente le proprie trasformazioni.
    """
    with Image.open(io.BytesIO(zf.read(path))) as source:
        rgb = source.convert("RGB")
        if crop is not None:
            x0, y0 = crop.left, crop.top
            x1, y1 = rgb.width - crop.right, rgb.height - crop.bottom
            if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
                raise ValueError(f"Crop impossibile per {path}: dimensione {rgb.size}, crop {crop}")
            rgb = rgb.crop((x0, y0, x1, y1))
        rgb = rgb.resize(image_size, resample=Image.Resampling.BILINEAR)
        array = np.array(rgb, dtype=np.uint8, copy=True)  # [H,W,C]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()  # [C,H,W]


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
        """Valida manifest, path, label e limiti PRIMA della messa in coda.

        Non decodifica i JPEG: il controllo completo di eventuali file immagine
        corrotti avviene comunque durante la preparazione delle finestre.
        """
        source = io.BytesIO(archive) if isinstance(archive, bytes) else archive
        with zipfile.ZipFile(source, "r") as zf:
            manifest = _read_manifest(zf, self.cfg)
            return {
                "episode_id": manifest["episode_id"],
                "frames": len(manifest["frames"]),
                "windows": self.count_windows(
                    len(manifest["frames"]),
                    window_size=self.cfg.window_size,
                    stride=self.cfg.stride,
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
            frames: list[dict[str, Any]] = manifest["frames"]
            episode_id: str = manifest["episode_id"]

            # Start 0, 5, 10, ...; le finestre sono SEMPRE contigue internamente.
            for window_i, start in enumerate(
                range(0, len(frames) - self.cfg.window_size + 1, self.cfg.stride)
            ):
                selected = frames[start : start + self.cfg.window_size]
                task = _prompt_for_window(selected)
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
    def count_windows(num_frames: int, *, window_size: int = 8, stride: int = 5) -> int:
        """Numero di finestre complete; gli ultimi frame residui non sono padded."""
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size e stride devono essere positivi")
        return 0 if num_frames < window_size else (num_frames - window_size) // stride + 1
