"""Salva episodi VLA-JEPA come PKL compatibili con script_controller_node.

Il PKL e' un dizionario {'traj': savers.Trajectory, ...metadati...},
creato dal metodo Trajectory.save() di dataset_collector_pkg/scripts/savers.py.

Le immagini sono array HWC BGR uint8, come nello script_controller. Il
sender recupera l'episodio dal PKL e genera lo ZIP solo in RAM per Flask.
ATTENZIONE: pickle.load() e' sicuro solo per file prodotti da fonti fidate.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vla_jepa_online.exploration.trajectory_recorder import RecordedEpisode, RecordedFrame


def _trajectory_type():
    """Importa la STESSA classe savers.Trajectory del controller ROS storico.

    Normalmente il package share di dataset_collector_pkg contiene scripts/.
    Nel preflight fuori da ROS si puo' rendere quel modulo visibile su PYTHONPATH.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        scripts = Path(get_package_share_directory('dataset_collector_pkg')) / 'scripts'
        if scripts.is_dir() and str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
    except Exception:
        # In preflight non-ROS il modulo puo' essere disponibile via PYTHONPATH.
        pass

    try:
        savers = importlib.import_module('savers')
    except ImportError as exc:
        raise RuntimeError(
            'savers.py non importabile. Eseguire nel container ROS con '
            'dataset_collector_pkg installato, oppure aggiungere '
            'dataset_collector_pkg/scripts a PYTHONPATH.'
        ) from exc

    Trajectory = savers.Trajectory
    if not callable(getattr(Trajectory, 'save', None)):
        raise RuntimeError(
            f'Savers errato: {savers.__file__}. Serve la versione di '
            'dataset_collector_pkg con Trajectory.save(path, **metadata), '
            'NON quella Multi-Task-LFD priva di save().' 
        )
    return Trajectory


def _decode_png_bgr(png: bytes, field: str) -> np.ndarray:
    if not isinstance(png, bytes) or not png.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError(f'{field}: PNG mancante/non valido')
    img = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f'{field}: impossibile decodificare il PNG RGB')
    return img  # OpenCV BGR, stesso formato del vecchio script_controller


def _encode_png_bgr(img: Any, field: str) -> bytes:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f'{field}: atteso array BGR uint8 [H,W,3]')
    ok, compressed = cv2.imencode(
        '.png', np.ascontiguousarray(arr),
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not ok:
        raise ValueError(f'{field}: codifica PNG fallita')
    return compressed.tobytes()


class TrajectorySaver:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError(f'Non e\' una directory: {self.root}')

    def save(
        self,
        episode_id: str,
        episode: RecordedEpisode,
        *,
        table_mode: str,
        waypoint_ids: tuple[int, ...],
        start_waypoint_id: int,
    ) -> Path:
        if not re.fullmatch(r'E\d{5,}', episode_id):
            raise ValueError(f'ID episodio inatteso: {episode_id!r}')
        if not episode.frames:
            raise ValueError('Non salvare un episodio vuoto')

        target = self.root / f'{episode_id}.pkl'
        if target.exists():
            raise FileExistsError(f'PKL gia\' presente: {target}')

        meta = {
            'schema_version': 1,
            'episode_id': episode_id,
            'table_mode': table_mode,
            'waypoint_ids': list(waypoint_ids),
            'start_waypoint_id': start_waypoint_id,
            'recording_fps_limit': episode.fps,
            'sampling_mode': 'motion_step',
            'ai_controller_target': 'vla_jepa_exploration_client',
        }
        Trajectory = _trajectory_type()
        trajectory = Trajectory(config_str=json.dumps(meta, separators=(',', ':')))

        for expected_index, frame in enumerate(episode.frames, start=1):
            if frame.index != expected_index:
                raise ValueError('Indici frame non consecutivi')
            if frame.robot_state is None or frame.commanded_action is None:
                raise ValueError(f'Frame {frame.index}: stato robot/azione mancanti')

            action = np.asarray(frame.commanded_action, dtype=np.float64)
            if action.shape != (8,) or not np.isfinite(action).all():
                raise ValueError(f'Frame {frame.index}: action 8D non valida')
            state = frame.robot_state

            # Il savers.py di dataset_collector_pkg comprime SOLO obs['image'].
            # Questi due array BGR non vengono ricompressi e sono direttamente
            # leggibili dai tool che analizzano i PKL di script_controller.
            obs = {
                'front_camera_image': _decode_png_bgr(frame.front_png, 'front_camera_image'),
                'gripper_camera_image': _decode_png_bgr(frame.gripper_png, 'gripper_camera_image'),
                'joint_pos': np.asarray(state['joint_pos'], dtype=np.float64),
                'joint_vel': np.asarray(state['joint_vel'], dtype=np.float64),
                'gripper_qpos': np.asarray(state['gripper_qpos'], dtype=np.float64),
                'gripper_qvel': np.asarray(state['gripper_qvel'], dtype=np.float64),
                'eef_pos': np.asarray(state['eef_pos'], dtype=np.float64),
                'eef_quat': np.asarray(state['eef_quat'], dtype=np.float64),
            }
            info = {
                'index': frame.index,
                'front_stamp_ns': frame.front_stamp_ns,
                'gripper_stamp_ns': frame.gripper_stamp_ns,
                'event': frame.event,
                'segment_id': frame.segment_id,
                'motion_label': frame.motion_label,
                'robot_state': state,
                'commanded_action': list(frame.commanded_action),
            }
            trajectory.append(
                obs=obs, action=action, info=info,
                reward=None, done=(expected_index == len(episode.frames)),
            )

        # Il metodo Trajectory.save() scrive direttamente nel path indicato.
        # Lo chiamiamo su un nome temporaneo, fsync, poi hardlink atomico:
        # il PKL visibile con ID definitivo non sara' mai parziale o sovrascritto.
        fd, temporary = tempfile.mkstemp(
            prefix=f'.{episode_id}.', suffix='.part', dir=self.root,
        )
        os.close(fd)
        try:
            trajectory.save(temporary, **meta)
            with open(temporary, 'rb') as handle:
                os.fsync(handle.fileno())
            os.link(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    @staticmethod
    def load_episode(path: str | Path) -> tuple[dict[str, Any], RecordedEpisode]:
        """Rilettura di un PKL fidato, per ricostruire lo ZIP in memoria."""
        import pickle

        path = Path(path)
        if path.suffix != '.pkl':
            raise ValueError(f'Atteso file .pkl, ricevuto {path}')
        Trajectory = _trajectory_type()  # import savers PRIMA di pickle.load()
        with path.open('rb') as handle:
            payload = pickle.load(handle)

        if not isinstance(payload, dict) or not isinstance(payload.get('traj'), Trajectory):
            raise ValueError('PKL incompatibile: atteso dizionario con chiave traj')
        traj = payload['traj']
        meta = json.loads(traj.config_str)
        if (meta.get('schema_version') != 1 or
                meta.get('sampling_mode') != 'motion_step' or
                not meta.get('episode_id') or
                len(traj) < 1):
            raise ValueError('Metadati PKL mancanti o incompatibili')
        # I metadati sono presenti sia a livello top-level sia in config_str.
        for key, value in meta.items():
            if payload.get(key) != value:
                raise ValueError(f'PKL: metadato incoerente: {key}')

        frames = []
        for idx in range(len(traj)):
            step = traj.get(idx, decompress=False)
            obs, info = step['obs'], step['info']
            if int(info['index']) != idx + 1:
                raise ValueError(f'Indice frame incoerente: {idx + 1}')
            action = np.asarray(step['action'], dtype=np.float64)
            commanded = tuple(info['commanded_action'])
            if (action.shape != (8,) or not np.isfinite(action).all()
                    or not np.array_equal(action, np.asarray(commanded, dtype=np.float64))):
                raise ValueError(f'Azione incoerente al frame {idx + 1}')
            frames.append(RecordedFrame(
                index=idx + 1,
                front_png=_encode_png_bgr(obs['front_camera_image'], 'front_camera_image'),
                gripper_png=_encode_png_bgr(obs['gripper_camera_image'], 'gripper_camera_image'),
                front_stamp_ns=int(info['front_stamp_ns']),
                gripper_stamp_ns=int(info['gripper_stamp_ns']),
                event=str(info['event']),
                segment_id=info.get('segment_id'),
                motion_label=info.get('motion_label'),
                robot_state=info.get('robot_state'),
                commanded_action=commanded,
            ))
        return meta, RecordedEpisode(
            fps=meta.get('recording_fps_limit'), frames=tuple(frames),
        )
