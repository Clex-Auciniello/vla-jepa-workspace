#!/usr/bin/env python3
"""Confronta la ricostruzione RGB dei latenti reali V-JEPA2 e dei latenti WM E4.

Usa una finestra robotica E4 di 8 frame (frontale e gripper), il checkpoint E4
completo e il decoder RGB Meta già addestrato. Non modifica i pesi.
Le tre predizioni sono teacher-forced e corrispondono ai tubelet futuri
all'interno della finestra, NON a un rollout oltre gli 8 frame.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

FRONT_KEY = "observation.images.exterior_1_left"
WRIST_KEY = "observation.images.exterior_2_left"
EXPECTED_OFFSETS = (1, 3, 5, 7)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-pkl", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0, help="Indice del primo frame, a partire da zero")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/E4/checkpoint_A"))
    parser.add_argument("--decoder", type=Path, default=Path(
        "/mnt/beegfs/a.cuciniello9/vjepa2_decoder/runs/meta_decoder_stage2/best.pt"))
    parser.add_argument("--decoder-code-dir", type=Path, default=Path(
        "/mnt/beegfs/a.cuciniello9/vjepa2_decoder"))
    parser.add_argument("--meta-repo", type=Path, default=None,
                        help="Root del repository Meta che contiene app/plan_common/models/wm_heads.py")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/e4_decoder_comparison"))
    parser.add_argument("--blur-sigma", type=float, default=0.0,
                        help="Sfocatura gaussiana solo per le immagini visualizzate; metriche sui pixel originali")
    args = parser.parse_args()
    if args.start < 0 or args.blur_sigma < 0:
        parser.error("start e blur-sigma devono essere >= 0")
    return args


def load_episode_window(path: Path, start: int):
    # I PKL robotici del framework sono serializzati con savers.Trajectory.
    importlib.import_module("savers")
    importlib.import_module("multi_task_il.datasets.savers")
    with path.open("rb") as stream:
        episode = pickle.load(stream)  # Solo PKL locali fidati.
    if not isinstance(episode, dict) or "traj" not in episode:
        raise ValueError("PKL senza traiettoria")
    traj = episode["traj"]
    if start + 8 > len(traj):
        raise ValueError(f"Servono otto frame da {start}, ma l'episodio ne ha {len(traj)}")
    frames = [traj.get(t, decompress=False)["obs"] for t in range(start, start + 8)]
    videos = []
    for key in ("camera_front_image", "camera_gripper_image"):
        sequence = []
        for step in frames:
            image = np.asarray(step[key])
            if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                raise ValueError(f"{key}: atteso RGB uint8 (224,224,3), ottenuto {image.shape}/{image.dtype}")
            sequence.append(np.ascontiguousarray(image))
        videos.append(np.stack(sequence))
    task = episode.get("language_instruction")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Il PKL E4 deve avere language_instruction non vuota")
    return videos[0], videos[1], task


def make_policy_batch(front: np.ndarray, wrist: np.ndarray, task: str, device):
    # Stessa conversione del batch_builder offline: RGB uint8 -> float [0,1].
    def tensor(images):
        return (torch.from_numpy(images.copy()).unsqueeze(0)
                .permute(0, 1, 4, 2, 3).contiguous().float().div_(255).to(device))
    return {FRONT_KEY: tensor(front), WRIST_KEY: tensor(wrist), "task": [task]}


def load_e4_policy(path: Path, device):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
    from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy

    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint E4 non trovato: {path}")
    cfg = PreTrainedConfig.from_pretrained(str(path))
    if not isinstance(cfg, VLAJEPAConfig) or not cfg.enable_world_model:
        raise ValueError("Il checkpoint non contiene il World Model VLA-JEPA")
    if int(cfg.num_video_frames) != 8 or int(cfg.jepa_tubelet_size) != 2 or cfg.num_world_model_views != 2:
        raise ValueError("Sono attesi 8 frame, tubelet_size=2 e due viste")
    if not hasattr(cfg, "causal_world_model_context"):
        raise RuntimeError("Questa revisione LeRobot non espone causal_world_model_context")
    if list(cfg.image_features.keys()) != [FRONT_KEY, WRIST_KEY]:
        raise ValueError(f"Ordine viste inatteso: {list(cfg.image_features)}. Evito uno split errato.")
    cfg.causal_world_model_context = True  # Stessa impostazione dei training offline E3/E4.
    cfg.device = str(device)
    policy = VLAJEPAPolicy.from_pretrained(str(path), config=cfg, strict=False)
    policy = policy.to(device).eval()
    return policy


def load_meta_decoder(checkpoint_path: Path, code_dir: Path, meta_repo: Path | None, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    # Il file di training può trovarsi nella root oppure nella sottocartella scripts/.
    candidates = [code_dir, code_dir / "scripts"]
    training_dir = next((p for p in candidates if (p / "train_meta_decoder.py").is_file()), None)
    if training_dir is None:
        raise FileNotFoundError(
            f"train_meta_decoder.py non trovato in {code_dir} né in {code_dir / 'scripts'}; "
            "usa --decoder-code-dir indicando la cartella corretta")
    sys.path.insert(0, str(training_dir.resolve()))
    roots = ([meta_repo] if meta_repo else []) + [
        code_dir, code_dir / "jepa-wms", code_dir / "jepa_wms",
        code_dir / "external" / "jepa-wms", code_dir / "vendor" / "jepa-wms",
        code_dir.parent / "jepa-wms",
    ]
    for root in roots:
        if root is not None and (root / "app" / "plan_common" / "models" / "wm_heads.py").is_file():
            sys.path.insert(0, str(root.resolve()))
            break
    # Riutilizziamo esattamente il costruttore Meta usato nel training.
    from train_meta_decoder import make_meta_head

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("tubelet_size") != 2 or checkpoint.get("frame_indices") != list(EXPECTED_OFFSETS):
        raise ValueError("Metadati temporali del decoder inattesi")
    head, head_cfg = make_meta_head(device)
    if checkpoint.get("head_cfg") != head_cfg:
        raise ValueError("La configurazione del decoder è differente da quella di best.pt")
    head.unwrapped.load_state_dict(checkpoint["decoder"], strict=True)
    head.unwrapped.eval()
    head.combined_loss.eval()
    return head, checkpoint


def capture_wm_prediction(policy, batch):
    # Il forward originale genera i token Qwen, applica il video processor,
    # codifica V-JEPA2 e invoca il predictor. Intercettiamo SOLO il suo output.
    result = {}
    def capture(_module, _inputs, output):
        if "predicted" in result:
            raise RuntimeError("Il predictor è stato chiamato più di una volta")
        if not isinstance(output, torch.Tensor):
            raise TypeError("Output del World Model non tensoriale")
        result["predicted"] = output.detach().float()
    handle = policy.model.video_predictor.register_forward_hook(capture)
    try:
        loss, logs = policy(batch)
    finally:
        handle.remove()
    if "predicted" not in result:
        raise RuntimeError("Il forward non ha eseguito il World Model")
    if not torch.isfinite(result["predicted"]).all():
        raise FloatingPointError("Il World Model ha prodotto embedding non finiti")
    return result["predicted"], float(loss.detach().float().item()), logs


def encode_oracle(policy, batch):
    # Stesso encoder e stesso processor del World Model E4, entrambi congelati.
    # Usiamo l'adapter LeRobot per ottenere le viste esattamente come nel training.
    inputs = policy._prepare_model_inputs(batch, training=True)
    videos = inputs["videos"]  # [1,2,8,3,224,224] in [0,1]
    if tuple(videos.shape) != (1, 2, 8, 3, 224, 224):
        raise ValueError(f"Shape video inattesa: {tuple(videos.shape)}")
    model = policy.model
    pixel = model.video_processor(
        videos=list(videos.reshape(2, 8, 3, 224, 224)),
        return_tensors="pt", device=model.video_encoder.device, do_rescale=False,
    )["pixel_values_videos"]
    if tuple(pixel.shape) != (2, 8, 3, 256, 256):
        raise ValueError(f"Il processor deve produrre [2,8,3,256,256], ottenuto {tuple(pixel.shape)}")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        encoded = model.video_encoder.get_vision_features(pixel_values_videos=pixel)
    # Encoder: 4 tubelet x 16x16 patch per camera x 1024 dimensioni.
    if tuple(encoded.shape) != (2, 4 * 256, 1024):
        raise ValueError(f"Latenti V-JEPA2 incompatibili: {tuple(encoded.shape)}")
    front_latents = encoded[0:1].reshape(1, 4, 256, 1024).float()
    # Ricostruiamo il target di decoder identico allo script Meta originale.
    mean = torch.as_tensor(model.video_processor.image_mean, device=pixel.device, dtype=pixel.dtype).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(model.video_processor.image_std, device=pixel.device, dtype=pixel.dtype).view(1, 1, 3, 1, 1)
    gt = (pixel[0:1].float() * std.float() + mean.float()).clamp(0, 1)
    gt = (gt[:, 1::2] * 255.0).round().byte().permute(0, 1, 3, 4, 2).cpu().numpy()[0]
    return front_latents, encoded.float(), gt


def to_uint8_decoder(head, embeddings):
    # La head Meta è stata addestrata su esattamente 4 posizioni temporali.
    if tuple(embeddings.shape) != (1, 4, 256, 1024):
        raise ValueError(f"Shape decoder inattesa: {tuple(embeddings.shape)}")
    features = embeddings.reshape(1, 4, 1, 16, 16, 1024)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoded = head.decode(features)
    if tuple(decoded.shape) != (1, 4, 1, 256, 256, 3) or decoded.dtype != torch.uint8:
        raise ValueError(f"Output Meta inatteso: {tuple(decoded.shape)} / {decoded.dtype}")
    return decoded[0, :, 0].cpu().numpy()


def mse_rgb(a, b):
    x = a.astype(np.float32) / 255.0
    y = b.astype(np.float32) / 255.0
    return float(np.mean((x - y) ** 2))


def visualize_image(rgb, sigma):
    return cv2.GaussianBlur(rgb, (3, 3), sigmaX=sigma, sigmaY=sigma) if sigma > 0 else rgb


def make_panel(frame_id, gt, oracle, pred, sigma):
    images = [gt, visualize_image(oracle, sigma), visualize_image(pred, sigma)]
    titles = [f"GT | frame {frame_id}", "DECODER(V-JEPA GT)", "DECODER(WM E4)"]
    width = gt.shape[1]
    bar = np.full((40, width * 3, 3), 26, dtype=np.uint8)
    for k, title in enumerate(titles):
        cv2.putText(bar, title, (k * width + 7, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.47, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, np.concatenate(images, axis=1)], axis=0)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Serve una GPU CUDA; lanciare con sbatch su gpuq")
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    front, wrist, task = load_episode_window(args.episode_pkl.resolve(), args.start)
    batch = make_policy_batch(front, wrist, task, device)
    print(f"Episodio: {args.episode_pkl} | frames {args.start}..{args.start+7}", flush=True)
    print(f"Prompt: {task}", flush=True)
    print(f"Checkpoint WM E4: {args.checkpoint}", flush=True)
    policy = load_e4_policy(args.checkpoint.resolve(), device)
    decoder, decoder_ckpt = load_meta_decoder(args.decoder.resolve(), args.decoder_code_dir.resolve(),
                                               args.meta_repo, device)
    print(f"Decoder: {args.decoder} (epoch {decoder_ckpt.get('epoch')})", flush=True)
    print(f"Encoder E4: {policy.config.jepa_encoder_name} | encoder decoder: {decoder_ckpt.get('encoder')}", flush=True)
    with torch.inference_mode():
        pred_all, forward_loss, logs = capture_wm_prediction(policy, batch)
        oracle_front, oracle_all, gt = encode_oracle(policy, batch)
        if tuple(pred_all.shape) != (1, 3 * 256, 2 * 1024):
            raise ValueError(f"Output WM inatteso: {tuple(pred_all.shape)}")
        # Encoder concatenato nell'ordine (frontale, gripper) lungo D.
        # Il target è lo shift temporale di una posizione: tubelet 1,2,3.
        target_all = torch.cat((oracle_all[0:1], oracle_all[1:2]), dim=-1)[:, 256:, :]
        latent_l1 = torch.nn.functional.l1_loss(pred_all, target_all).item()
        predicted_front = pred_all.reshape(1, 3, 256, 2048)[..., :1024]
        latent_front_l1 = torch.nn.functional.l1_loss(predicted_front, oracle_front[:, 1:]).item()
        # Ripristiniamo quattro posizioni, come richiesto dal decoder addestrato.
        hybrid = torch.cat([oracle_front[:, :1], predicted_front], dim=1)
        oracle_frames = to_uint8_decoder(decoder, oracle_front)
        wm_frames = to_uint8_decoder(decoder, hybrid)
    if abs(latent_l1 - forward_loss) > 2e-3:
        print(f"ATTENZIONE: WM loss forward={forward_loss:.6f}, ricomputata={latent_l1:.6f}. "
              "Verificare impostazione causale/processor locale.", flush=True)
    rows, panels = [], []
    for t in (1, 2, 3):
        frame_id = args.start + EXPECTED_OFFSETS[t]
        ref, oracle, wm = gt[t], oracle_frames[t], wm_frames[t]
        panel = make_panel(frame_id, ref, oracle, wm, args.blur_sigma)
        panels.append(panel)
        Image.fromarray(panel, mode="RGB").save(args.output_dir / f"frame_{frame_id:04d}.png")
        rows.append({"frame_id_zero_based": frame_id,
                     "gt_oracle_mse": mse_rgb(ref, oracle),
                     "gt_wm_mse": mse_rgb(ref, wm),
                     "oracle_wm_mse": mse_rgb(oracle, wm)})
    Image.fromarray(np.concatenate(panels, axis=0), mode="RGB").save(args.output_dir / "comparison.png")
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "episode": str(args.episode_pkl.resolve()), "start": args.start,
        "world_model_checkpoint": str(args.checkpoint.resolve()),
        "decoder_checkpoint": str(args.decoder.resolve()),
        "decoder_encoder": decoder_ckpt.get("encoder"),
        "world_model_encoder": policy.config.jepa_encoder_name,
        "prompt": task, "causal_world_model_context": True,
        "wm_loss_forward": forward_loss, "wm_latent_l1_recomputed": latent_l1,
        "wm_front_latent_l1": latent_front_l1,
        "mean_oracle_mse_rgb01": float(np.mean([r["gt_oracle_mse"] for r in rows])),
        "mean_world_model_mse_rgb01": float(np.mean([r["gt_wm_mse"] for r in rows])),
        "target_frame_indices_zero_based": [r["frame_id_zero_based"] for r in rows],
        "blur_sigma_display_only": args.blur_sigma,
        "note": "Predizioni teacher-forced interne alla finestra di 8 frame, NON rollout oltre la finestra",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Risultati salvati in: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
