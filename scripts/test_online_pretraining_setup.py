#!/usr/bin/env python3
"""Preflight offline del pretraining online VLA-JEPA / UR5e.

Pensato per essere collocato in <workspace>/scripts/test_online_pretraining_setup.py.
Non importa ROS, non apre camere, non invia comandi al robot e non avvia un
server HTTP in ascolto. Flask viene verificato tramite test_client in-process.

Livelli:
  --mode quick   controlli CPU + kernel CUDA leggero (NO caricamento Qwen)
  --mode gpu     quick + caricamento VLA-JEPA + UN forward/backward reale
  --optimizer-step  [con --mode gpu] 9 ulteriori finestre + AdamW.step()
  --checkpoint-save [con --optimizer-step] snapshot reale in directory TEMP
  --resume-test  [con --checkpoint-save] ricarica pesi + optimizer

I test GPU modificano parametri SOLO IN RAM. L'opzione checkpoint-save scrive
un checkpoint di test in un percorso dedicato e lo elimina a fine esecuzione.
Non usare --checkpoint-root del training reale come directory di test.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import io
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path


def log(*parts):
    print(*parts, flush=True)


def ensure(cond, message):
    if not cond:
        raise AssertionError(message)


def section(name):
    log('\n' + '=' * 76)
    log(name)
    log('=' * 76)


def load_workspace(args):
    root = Path(args.workspace).expanduser().resolve() if args.workspace else Path(__file__).resolve().parents[1]
    source = root / 'src'
    ensure((source / 'vla_jepa_online' / 'training_server' / 'online_trainer.py').is_file(),
           f'Workspace non riconosciuto: {root} (manca src/vla_jepa_online/...)')
    sys.path.insert(0, str(source))
    return root


def check_environment(root):
    import numpy
    import torch
    import cv2
    import PIL
    import yaml

    section('1 - Ambiente Python, sorgenti e CUDA')
    log('Python:', sys.executable, '|', sys.version.split()[0])
    log('Architettura:', platform.machine())
    log('Venv:', sys.prefix, '| base:', sys.base_prefix)
    ensure(sys.prefix != sys.base_prefix, 'Eseguire con il Python del venv, non con quello del container ROS')
    for dist in ('torch', 'torchvision', 'transformers', 'lerobot', 'Flask',
                 'numpy', 'Pillow', 'opencv-python', 'safetensors', 'accelerate', 'draccus'):
        log(f'{dist:22s}', metadata.version(dist))
    ensure(sys.version_info[:2] == (3, 12), 'Il setup Spark atteso usa Python 3.12')

    import lerobot
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
    le_path = Path(inspect.getfile(VLAJEPAPolicy)).resolve()
    log('LeRobot effettivo:', le_path)
    train_repo = root / 'external' / 'lerobot'
    inferred_repo = le_path.parents[4]  # .../lerobot/src/lerobot/policies/vla_jepa/file.py -> repo
    # Confronto reale delle revisioni: lo stesso commit non esclude modifiche locali.
    for title, repo in [('Training', train_repo), ('Importato', inferred_repo)]:
        if (repo / '.git').exists():
            commit_cmd = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'],
                                        capture_output=True, text=True)
            if commit_cmd.returncode == 0:
                dirty_cmd = subprocess.run(['git', '-C', str(repo), 'status', '--porcelain'],
                                           capture_output=True, text=True)
                log(f'{title} repo:', commit_cmd.stdout.strip(), '| working tree:',
                    ('MODIFICATO' if dirty_cmd.stdout.strip() else 'pulito')
                    if dirty_cmd.returncode == 0 else 'non verificabile')
            else:
                log(f'{title} repo: Git non risolvibile nel container (bind mount parziale).')
                log('  Dettaglio:', commit_cmd.stderr.strip()[:250])
        else:
            log(f'{title} repo:', repo, '(git non disponibile)')

    log('Torch CUDA:', torch.version.cuda, '| GPU:', torch.cuda.is_available())
    ensure(torch.cuda.is_available(), 'CUDA non disponibile nel container')
    log('Dispositivo:', torch.cuda.get_device_name(0), '| capability:', torch.cuda.get_device_capability(0))
    ensure(torch.cuda.is_bf16_supported(), 'BF16 non supportato')
    free, total = torch.cuda.mem_get_info()
    log(f'Memoria CUDA: libera {free/2**30:.2f} GiB / totale {total/2**30:.2f} GiB')
    # Un'operazione FP/BF16 reale controlla molto piu' di torch.cuda.is_available.
    a = torch.randn(256, 256, device='cuda', dtype=torch.bfloat16)
    out = (a @ a).float().square().mean()
    ensure(bool(torch.isfinite(out).item()), 'Matmul BF16 CUDA con output non finito')
    torch.cuda.synchronize()
    del a, out
    torch.cuda.empty_cache()
    r = torch.linspace(.99, 1.01, steps=4096, device='cuda', dtype=torch.float32).prod()
    ensure(bool(torch.isfinite(r).item()), 'Riduzione CUDA/NVRTC non finita')
    torch.cuda.synchronize()
    del r
    log('BF16 matmul / CUDA reduction: OK')
    log('Nota: un test CUDA leggero NON prova il funzionamento di tutti i kernel del backward.')


def check_checkpoint(root, path):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
    from vla_jepa_online.training_server.online_trainer import OnlineVideoTrainer
    from vla_jepa_online.training_server.online_batch_builder import OnlineBatchBuilderConfig

    section('2 - Configurazione e file checkpoint DROID')
    ensure(path.is_dir(), f'Checkpoint assente: {path}')
    ensure((path / 'config.json').is_file(), f'config.json assente: {path}')
    cfg = PreTrainedConfig.from_pretrained(str(path))
    log('Checkpoint:', path)
    log('Classe:', type(cfg).__name__, '| type:', cfg.type)
    ensure(isinstance(cfg, VLAJEPAConfig), 'Il checkpoint non contiene VLAJEPAConfig')
    ensure(cfg.type == 'vla_jepa', 'Policy type differente da vla_jepa')
    for key, expected in [('enable_world_model', True), ('freeze_qwen', False),
                          ('use_peft', False), ('num_video_frames', 8), ('jepa_tubelet_size', 2)]:
        value = getattr(cfg, key)
        log(f'{key}: {value}')
        ensure(value == expected, f'{key}: richiesto {expected}, trovato {value}')
    features = cfg.image_features
    builder_cfg = OnlineBatchBuilderConfig()
    expected_keys = {builder_cfg.front_feature, builder_cfg.gripper_feature}
    ensure(set(features) == expected_keys,
           f'Feature checkpoint {list(features)} diverse dal batch builder {sorted(expected_keys)}')
    for key, feature in features.items():
        log('Image feature:', key, feature.shape)
        ensure(tuple(feature.shape) == (3, 224, 224), f'Shape errata {key}')
    log('Optimizer:', cfg.optimizer_betas, cfg.optimizer_eps, cfg.optimizer_weight_decay)
    # Controllo che il codice corrente sia stato DAVVERO corretto, non solo il test esterno.
    init_source = inspect.getsource(OnlineVideoTrainer.__init__)
    ensure('PreTrainedConfig.from_pretrained(' in init_source,
           'Il trainer usa ancora la deserializzazione diretta VLAJEPAConfig.from_pretrained: correggere online_trainer.py')
    ensure('VLAJEPAPolicy.from_pretrained(' in init_source, 'Caricamento policy assente')
    safes = sorted(path.glob('*.safetensors'))
    indexes = sorted(path.glob('*.safetensors.index.json'))
    ensure(safes or indexes, 'Pesi safetensors non trovati nella directory del checkpoint')
    for p in safes:
        log(f'Pesi: {p.name} ({p.stat().st_size/2**30:.2f} GiB)')
    if indexes:
        log('Index safetensors:', ', '.join(p.name for p in indexes))
    log('Configurazione checkpoint / contratto batch: OK')
    return cfg


def check_cache():
    section('3 - Cache Hugging Face (diagnostica offline)')
    hf_home = Path(os.environ.get('HF_HOME', '~/.cache/huggingface')).expanduser()
    hub = Path(os.environ.get('HF_HUB_CACHE', str(hf_home / 'hub'))).expanduser()
    log('HF_HOME:', hf_home, '| presente:', hf_home.is_dir())
    log('HF_HUB_CACHE:', hub, '| presente:', hub.is_dir())
    for repo in ('models--Qwen--Qwen3-VL-2B-Instruct', 'models--facebook--vjepa2-vitl-fpc64-256'):
        r = hub / repo
        log(repo, '| presente:', r.is_dir())
    log('Nota: i nomi della cache sono indicativi; il caricamento GPU verifica le dipendenze effettive.')


def _make_synthetic_grid():
    # Usato quando il workspace_grid.json reale non e' ancora stato misurato.
    vertices = {}
    for iz in (0, 1):
        for iy in (0, 1):
            for ix in (0, 1):
                vertices[f'V{ix}{iy}{iz}'] = [-0.3 + ix * 0.255, 0.20 + iy * 0.255, 0.11 + iz * 0.17]
    offsets = {'x': [0., .085, .170, .255],
               'y': [0., .085, .170, .255], 'z': [0., .085, .170]}
    waypoints = []
    for k, z in enumerate(offsets['z']):
        for j, y in enumerate(offsets['y']):
            for i, x in enumerate(offsets['x']):
                waypoints.append({'id': len(waypoints), 'i': i, 'j': j, 'k': k,
                                  'position': [-0.3 + x, .20 + y, .11 + z],
                                  'level': 'PICK' if k == 0 else 'TRANSFER'})
    return {'base_frame': 'base_link', 'tcp_frame': 'tcp_link',
            'home': [0, 0, 0, 0, 0, 0, 1], 'vertices': vertices,
            'dimensions_m': {'x': .255, 'y': .255, 'z': .17},
            'sampling': {'axis_offsets_m': offsets},
            'grid_shape': {'nx': 4, 'ny': 4, 'nz': 3},
            'num_waypoints': len(waypoints), 'waypoints': waypoints}


class FakeRobotBackend:
    """Backend senza connessioni ROS, senza controllo robot."""
    def __init__(self):
        self.pose = None
        self.commands = []

    def move_home(self):
        self.commands.append(('home', None))

    def set_gripper(self, closed):
        self.commands.append(('gripper', bool(closed)))

    def move_to_pose(self, pose):
        self.pose = pose
        self.commands.append(('pose', pose))

    def get_tcp_pose(self):
        return self.pose


def check_exploration(root):
    from vla_jepa_online.exploration.coverage_tracker import CoverageTracker
    from vla_jepa_online.exploration.random_walker import RandomWalker, load_exploration_seed
    from vla_jepa_online.exploration.motion_sampler import MotionSampler
    from vla_jepa_online.exploration.robot_executor import RobotExecutor, RobotExecutorConfig
    from vla_jepa_online.exploration.trajectory_recorder import TrajectoryRecorderConfig
    from vla_jepa_online.exploration.exploration_client import ExplorationClientConfig
    from vla_jepa_online.utils.geometry_utils import build_linear_waypoints

    section('4 - Config YAML, griglia, random walk, motion sampler, executor simulato')
    config_file = root / 'configs' / 'exploration.yaml'
    ensure(config_file.is_file(), f'Manca {config_file}')
    seed = load_exploration_seed(config_file)
    walker = RandomWalker.from_yaml(config_file)
    sampler = MotionSampler.from_yaml(config_file)
    robot_cfg = RobotExecutorConfig.from_yaml(config_file)
    TrajectoryRecorderConfig.from_yaml(config_file)
    client_cfg = ExplorationClientConfig.from_yaml(config_file)
    ensure(client_cfg.min_frames <= 8, 'min_frames del client > 8, incoerente con la minima finestra')
    real_grid_path = root / 'configs' / 'workspace_grid.json'
    grid = json.loads(real_grid_path.read_text()) if real_grid_path.is_file() else _make_synthetic_grid()
    log('Griglia:', str(real_grid_path) if real_grid_path.is_file() else 'SINTETICA (griglia reale assente)')
    log('Waypoints:', len(grid['waypoints']), '| seed:', seed,
        '| waypoint per traiettoria:', walker.cfg.num_waypoints)
    ensure(len(grid['waypoints']) >= walker.cfg.num_waypoints + 1,
           'Griglia insufficiente per START e waypoint distinti')
    ensure({w['level'] for w in grid['waypoints']} == {'PICK', 'TRANSFER'}, 'Servono livelli PICK/TRANSFER')

    with tempfile.TemporaryDirectory(prefix='jepa_preflight_tracker_') as temp_dir:
        state_path = Path(temp_dir) / 'coverage.json'
        tracker = CoverageTracker(grid, state_path, seed=seed)
        ensure(tracker.next_episode_id == 'E00001', 'ID iniziale non valido')
        total_motions = 0
        for mode in ('empty', 'occupied'):
            start = walker.sample_start(grid, tracker.start_visits, rng=tracker.rng)
            walk = walker.sample(grid, tracker.accepted_visits, table_mode=mode,
                                 rng=tracker.rng, exclude_id=start)
            ensure(len(walk.waypoint_ids) == walker.cfg.num_waypoints, 'Numero waypoint errato')
            ensure(len(set(walk.waypoint_ids)) == len(walk.waypoint_ids), 'Waypoint duplicati')
            ensure(start not in walk.waypoint_ids, 'START incluso fra waypoint esplorativi')
            motion = sampler.sample(grid, walk.waypoint_ids, table_mode=mode, rng=tracker.rng)
            rotations = sum(step.rotate_here for step in motion.steps)
            closes = sum(step.toggle_gripper_here and step.gripper_closed_after for step in motion.steps)
            opens = sum(step.toggle_gripper_here and not step.gripper_closed_after for step in motion.steps)
            ensure(rotations <= 2 and closes <= 1 and opens <= 1, 'Eventi robot fuori limite')
            fake = FakeRobotBackend()
            executor = RobotExecutor(fake, robot_cfg)
            first = next(w for w in grid['waypoints'] if w['id'] == start)
            start_pose = executor.move_to_start(first)
            plan = executor.build_commands(motion, start_pose, grid=grid, start_waypoint=first)
            callbacks = []
            executor.execute(plan, on_step=lambda event, **info: callbacks.append((event, info)))
            ensure(len(callbacks) >= len(walk.waypoint_ids), 'Micro-step non eseguiti dal backend finto')
            ensure(all(c[1].get('motion_label') for c in callbacks),
                   'Una cattura simulata non ha motion_label: il batch builder potrebbe rifiutare il video')
            total_motions += len(callbacks)
            tracker.accept_episode(tracker.next_episode_id, plan.waypoint_ids, start)
        ensure(tracker.next_episode_number == 3, 'Contatore episodi errato')
        restored = CoverageTracker(grid, state_path, seed=seed)
        ensure(restored.next_episode_id == 'E00003', 'Resume tracker fallito')
        ensure(restored.accepted_visits == tracker.accepted_visits, 'Visite non persistite')
        ensure(restored.rng.getstate() == tracker.rng.getstate(), 'Stato RNG non persistito')
        restored.discard_episode(restored.next_episode_id)
        ensure(restored.next_episode_id == 'E00003', 'Discard ha consumato ID')
        log('Tracker persist/reload/discard: OK; micro-step simulati:', total_motions)

    # Comportamento implementato oggi (floor), NON confondere con un limite <=2cm.
    waypoints = build_linear_waypoints([0, 0, 0], [0, 0, 0, 1], [.051, 0, 0], [0, 0, 0, 1], .02)
    max_delta = max(math.dist([0, 0, 0] if i == 0 else waypoints[i-1].position, w.position)
                    for i, w in enumerate(waypoints))
    if max_delta > .02 + 1e-7:
        log(f'WARNING geometria: micro-step massimo nell\'esempio {max_delta*100:.2f} cm (>2 cm).')
        log('  compute_num_steps usa floor, quindi 2 cm NON e\' un limite superiore.')
    log('Esplorazione completamente simulata: OK (nessun import ROS, nessun comando reale)')


def synthetic_episode_payload(episode_id='PREFLIGHT_001', frames_count=13):
    """Usa i dataclass reali del recorder e l'encoder ZIP reale del client."""
    import numpy as np
    import cv2
    from vla_jepa_online.exploration.trajectory_recorder import RecordedFrame, RecordedEpisode
    from vla_jepa_online.exploration.exploration_client import UploadTask, FlaskTrainingTransport

    # Frame RGB distinti e ripetibili, non immagini nere uniformi.
    h, w = 376, 672
    xs = np.arange(w, dtype=np.uint8)[None, :]
    ys = np.arange(h, dtype=np.uint8)[:, None]
    frames = []
    for idx in range(1, frames_count + 1):
        a = np.zeros((h, w, 3), dtype=np.uint8)
        a[:, :, 0] = (xs + idx * 5).astype(np.uint8)
        a[:, :, 1] = (ys + idx * 3).astype(np.uint8)
        a[:, :, 2] = (idx * 17) % 256
        b = a[:, ::-1, :].copy()
        ok1, f = cv2.imencode('.png', cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        ok2, g = cv2.imencode('.png', cv2.cvtColor(b, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        ensure(ok1 and ok2, 'Encoding PNG sintetico fallito')
        label = 'Move the gripper right' if idx <= 5 else ('Close the gripper' if idx == 6 else 'Move the gripper up')
        frames.append(RecordedFrame(index=idx, front_png=f.tobytes(), gripper_png=g.tobytes(),
                                    front_stamp_ns=10**12 + idx * 100_000_000,
                                    gripper_stamp_ns=10**12 + idx * 100_000_000 + 1_000_000,
                                    event='start' if idx == 1 else 'move',
                                    segment_id=1, motion_label=label))
    episode = RecordedEpisode(fps=None, frames=tuple(frames))
    return FlaskTrainingTransport.encode(UploadTask(episode_id=episode_id, table_mode='empty',
                                                     waypoint_ids=(1,2,3,4,5),
                                                     start_waypoint_id=0, episode=episode))


def check_data_pipeline():
    import torch
    from vla_jepa_online.training_server.online_batch_builder import OnlineBatchBuilder
    section('5 - ZIP client, manifest, finestre video, crop e prompt')
    payload = synthetic_episode_payload()
    builder = OnlineBatchBuilder()
    info = builder.inspect_episode(payload)
    ensure(info['frames'] == 13 and info['windows'] == 2,
           f'Conteggio finestre scorretto: {info}')
    windows = list(builder.iter_episode_windows(payload))
    ensure([w.start_index for w in windows] == [0, 5], 'Stride 5 errato')
    for win in windows:
        ensure(set(win.batch) == {builder.cfg.front_feature, builder.cfg.gripper_feature, 'task'},
               'Campi batch inattesi')
        for key in (builder.cfg.front_feature, builder.cfg.gripper_feature):
            t = win.batch[key]
            ensure(t.shape == (1,8,3,224,224), f'{key}: shape {t.shape}')
            ensure(t.dtype == torch.float32 and bool(torch.isfinite(t).all()), f'{key}: dtype/finite')
            ensure(0 <= float(t.min()) and float(t.max()) <= 1, f'{key}: fuori range [0,1]')
        ensure(win.batch['task'][0], 'Prompt vuoto')
    log('ZIP:', len(payload), 'byte; frames:', info['frames'], '; windows:', info['windows'])
    for win in windows:
        log('Finestra', win.window_index, 'start', win.start_index, '| task:', win.batch['task'][0])
    log('Coppie PNG reali -> tensori [1,8,3,224,224] float32 -> prompt: OK')
    return payload, windows[0].batch


def check_server_mock(payload):
    from vla_jepa_online.training_server.server import TrainingService, ServerConfig, create_app
    section('6 - Endpoint Flask / FIFO con trainer finto (nessuna GPU di training)')

    class FakeTrainer:
        global_step = 0
        processed_videos = 0
        pending_windows = 0

        def train_episode(self, episode_id, batches, on_window_complete=None):
            count = 0
            for batch in batches:
                count += 1
                ensure('task' in batch, 'Batch server privo di task')
                if on_window_complete:
                    on_window_complete(count, 0.125)
            self.processed_videos += 1
            self.global_step += 1
            return {'episode_id': episode_id, 'windows': count, 'global_step': self.global_step,
                    'processed_videos': self.processed_videos, 'pending_windows': 0,
                    'mean_wm_loss': .125, 'checkpoint_path': None}

    with tempfile.TemporaryDirectory(prefix='jepa_preflight_flask_') as folder:
        service = TrainingService(ServerConfig(checkpoint_root=folder, max_queued_episodes=3),
                                  trainer_factory=lambda resume: FakeTrainer())
        try:
            app = create_app(service)
            client = app.test_client()  # nessuna socket aperta, nessun bind HTTP
            deadline = time.monotonic() + 10
            while service.status()['server'] == 'initializing' and time.monotonic() < deadline:
                time.sleep(.025)
            ensure(service.status()['server'] == 'ready', f'Server fake non pronto: {service.status()}')
            bad = client.post('/train_episode', data=b'junk', content_type='text/plain')
            ensure(bad.status_code == 415, f'Content-Type errato accettato: {bad.status_code}')
            badzip = client.post('/train_episode', data=b'not-a-zip', content_type='application/zip')
            ensure(badzip.status_code == 400, f'ZIP non valido accettato: {badzip.status_code}')
            mismatch = client.post('/train_episode', data=payload,
                                   content_type='application/zip', headers={'X-Episode-ID':'WRONG_ID'})
            ensure(mismatch.status_code == 400, f'Header ID errato accettato: {mismatch.status_code}')
            response = client.post('/train_episode', data=payload,
                                   content_type='application/zip', headers={'X-Episode-ID': 'PREFLIGHT_001'})
            ensure(response.status_code == 202, f'POST non accettato: {response.status_code} {response.json}')
            duplicate = client.post('/train_episode', data=payload,
                                    content_type='application/zip', headers={'X-Episode-ID': 'PREFLIGHT_001'})
            ensure(duplicate.status_code == 200, f'Duplicato non riconosciuto: {duplicate.status_code}')
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                status = client.get('/episodes/PREFLIGHT_001').json
                if status['status'] in ('trained', 'failed'):
                    break
                time.sleep(.03)
            ensure(status['status'] == 'trained', f'Trainer fake non completo: {status}')
            ensure(status['windows_done'] == 2, f'Avanzamento finestre errato: {status}')
            st = client.get('/status').json
            ensure(st['trainer']['processed_videos'] == 1, f'Contatori errati: {st}')
            ensure(client.get('/queue').status_code == 200, 'GET /queue fallito')
            ensure(st['safe_to_stop'] is False, 'safe_to_stop non deve essere true senza checkpoint')
            log('POST: 202 queued; duplicato: 200; ID errato: 400; Content-Type: 415')
            log('GET /status, /queue, /episodes/<id>: OK; worker FIFO fake: trained')
        finally:
            service.close()
    log('Server Flask test_client: OK (nessun server reale avviato)')


def _gradient_status(parameters):
    grads = [p.grad for p in parameters if p.grad is not None]
    any_nonzero = any(bool(torch_grad.detach().abs().max().item() > 0) for torch_grad in grads[:64])
    all_finite = all(bool(g.detach().isfinite().all().item()) for g in grads[:64])
    return len(grads), any_nonzero, all_finite


def check_gpu_training(checkpoint, batch, args):
    import torch
    from vla_jepa_online.training_server.online_trainer import OnlineVideoTrainer, OnlineTrainerConfig
    section('7 - Training GPU REALE: modello DROID, WM loss, backward')
    log('Questo test puo\' usare molta RAM/GPU; non avvia ROS o robot.')
    start = time.monotonic()
    parent = None
    if args.checkpoint_save:
        parent = Path(args.checkpoint_test_root).expanduser().resolve()
        ensure(parent.is_dir(), f'Directory checkpoint-test non presente: {parent}')
    with tempfile.TemporaryDirectory(prefix='jepa_preflight_gpu_', dir=parent) as tmp:
        trainer = OnlineVideoTrainer(OnlineTrainerConfig(pretrained_path=str(checkpoint),
                                                         checkpoint_root=tmp,
                                                         qwen_gradient_checkpointing=True))
        log('Modello e optimizer inizializzati in', round(time.monotonic()-start, 1), 's')
        log('Parametri addestrabili:', trainer.num_trainable_parameters)
        mod = trainer.policy.model
        ensure(all(not p.requires_grad for p in mod.video_encoder.parameters()), 'V-JEPA non frozen')
        ensure(all(not p.requires_grad for p in mod.action_model.parameters()), 'Action Head non frozen')
        ensure(any(p.requires_grad for p in mod.qwen.parameters()), 'Qwen frozen')
        ensure(any(p.requires_grad for p in mod.video_predictor.parameters()), 'Predictor frozen')
        result = trainer.train_episode('PREFLIGHT_GPU_001', [batch])
        log('Risultato primo backward:', result)
        ensure(math.isfinite(result['mean_wm_loss']), 'WM loss NaN/inf')
        ensure(trainer.pending_windows == 1 and trainer.global_step == 0,
               'Con una finestra devono rimanere gradienti in accumulo; no optimizer step')
        n_q, nz_q, fin_q = _gradient_status(mod.qwen.parameters())
        n_p, nz_p, fin_p = _gradient_status(mod.video_predictor.parameters())
        ensure(n_q > 0 and nz_q and fin_q, f'Gradienti Qwen mancanti/non finiti: {(n_q,nz_q,fin_q)}')
        ensure(n_p > 0 and nz_p and fin_p, f'Gradienti Predictor mancanti/non finiti: {(n_p,nz_p,fin_p)}')
        ensure(all(p.grad is None for p in mod.video_encoder.parameters()), 'Gradienti inattesi su V-JEPA')
        ensure(all(p.grad is None for p in mod.action_model.parameters()), 'Gradienti inattesi su Action Head')
        log('Qwen gradients:', n_q, 'tensori, nonzero:', nz_q, 'finite:', fin_q)
        log('Predictor gradients:', n_p, 'tensori, nonzero:', nz_p, 'finite:', fin_p)
        log('CUDA max allocated:', round(torch.cuda.max_memory_allocated()/2**30,2), 'GiB')

        if args.optimizer_step:
            section('8 - Nove finestre aggiuntive e un aggiornamento AdamW effettivo')
            # A(1)+B(9) -> accumulo completo 10 -> 1 optimizer.step().
            more = trainer.train_episode('PREFLIGHT_GPU_002', (batch for _ in range(9)))
            log('Risultato secondo episodio:', more)
            ensure(trainer.global_step == 1 and trainer.pending_windows == 0,
                   'AdamW.step() non effettuato esattamente dopo 10 finestre')
            ensure(more['optimizer_steps_for_video'] == 1, 'Numero optimizer step inatteso')
            ensure(more['last_grad_norm'] is not None and math.isfinite(more['last_grad_norm']),
                   'Grad norm non finita')
            log('AdamW.step(): OK; grad_norm:', more['last_grad_norm'])

        if args.checkpoint_save:
            section('9 - Salvataggio checkpoint REALE (temporaneo, scrittura intensiva)')
            saved = trainer.save_checkpoint()
            log('Salvato:', saved)
            for fname in ('config.json', 'model.safetensors',
                          'online_trainer_state.pt', 'checkpoint_meta.json'):
                ensure((saved / fname).is_file(), f'Checkpoint incompleto: {fname}')
                log(' -', fname, f'{(saved / fname).stat().st_size/2**30:.2f} GiB')
            from vla_jepa_online.training_server.server import _checkpoint_latest
            ensure(_checkpoint_latest(Path(tmp)) == saved, 'Selezione ultimo checkpoint fallita')
            if args.resume_test:
                section('10 - Resume REALE checkpoint e optimizer')
                # Libera il modello precedente prima di ripristinare il successivo.
                expected_steps, expected_videos = trainer.global_step, trainer.processed_videos
                del mod, trainer
                gc.collect()
                torch.cuda.empty_cache()
                resumed = OnlineVideoTrainer(OnlineTrainerConfig(
                    pretrained_path=str(checkpoint), resume_from=saved,
                    checkpoint_root=tmp))
                ensure(resumed.global_step == expected_steps, 'Resume global_step errato')
                ensure(resumed.processed_videos == expected_videos, 'Resume processed_videos errato')
                log('Resume pesi e optimizer: OK', resumed.global_step, resumed.processed_videos)
                del resumed
            log('Checkpoint temporaneo: OK, la directory viene rimossa automaticamente')

    gc.collect()
    torch.cuda.empty_cache()
    log('TRAINING PREFLIGHT GPU PASSED in', round(time.monotonic()-start,1), 's')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workspace', help='Workspace root; default: parent della cartella scripts')
    parser.add_argument('--checkpoint', default='checkpoints/base',
                        help='Directory del pretrained_model DROID, relativa al workspace')
    parser.add_argument('--mode', choices=('quick','gpu'), default='quick',
                        help='quick=no training; gpu=1 forward/backward reale')
    parser.add_argument('--optimizer-step', action='store_true',
                        help='Richiede --mode gpu; completa 10 backward + AdamW.step')
    parser.add_argument('--checkpoint-save', action='store_true',
                        help='Richiede --optimizer-step; salva un checkpoint GPU temporaneo')
    parser.add_argument('--checkpoint-test-root',
                        help='Directory SU DISCO con molto spazio libero, usata solo per il checkpoint di test')
    parser.add_argument('--resume-test', action='store_true',
                        help='Richiede --checkpoint-save; ricarica modello e AdamW dallo snapshot temporaneo')
    args = parser.parse_args()
    if args.optimizer_step and args.mode != 'gpu':
        parser.error('--optimizer-step richiede --mode gpu')
    if args.checkpoint_save and (not args.optimizer_step or not args.checkpoint_test_root):
        parser.error('--checkpoint-save richiede --optimizer-step e --checkpoint-test-root')
    if args.resume_test and not args.checkpoint_save:
        parser.error('--resume-test richiede --checkpoint-save')

    # Previene download impliciti nei test: il controllo offline deve trovare tutto in cache.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    root = load_workspace(args)
    checkpoint = Path(args.checkpoint).expanduser()
    checkpoint = (root / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
    section('VLA-JEPA ONLINE - PREFLIGHT SENZA ROBOT')
    log('Workspace:', root, '\nCheckpoint:', checkpoint, '\nModalita:', args.mode)
    check_environment(root)
    check_checkpoint(root, checkpoint)
    check_cache()
    check_exploration(root)
    payload, batch = check_data_pipeline()
    check_server_mock(payload)
    if args.mode == 'gpu':
        check_gpu_training(checkpoint, batch, args)
    section('RISULTATO')
    log('PREFLIGHT SUPERATO:', 'GPU training effettivo' if args.mode == 'gpu' else 'controlli leggeri')
    log('Nessun comando ROS/MoveIt/Robotiq e nessuna registrazione camera eseguiti.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        log('\nPREFLIGHT FALLITO:', type(exc).__name__, str(exc))
        sys.exit(1)
