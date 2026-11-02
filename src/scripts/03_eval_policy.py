"""
03_eval_policy.py
=================
Evaluation script comparing trained SAC policy against baseline nominal Bézier trajectories.
Computes:
- Success Rate (SR) with 95% Wilson Score confidence intervals.
- Final Euclidean Distance to Target (m).
- Alignment Error (m) and Table Contact Ratio (%).
- Episode Returns and execution times.
- Optional MP4 Video Recording of Best Attempt, Worst Attempt, and Nominal Baseline.
Outputs formatted comparative summary, JSON reports, and video artifacts.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

# Add project root to sys.path
WS_ROOT = Path(__file__).resolve().parent.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from envs.dynamic_folding_env import DynamicFoldingEnv
from models.sac_policy import SACAgent
from callbacks.curriculum_callback import CurriculumManager


def compute_wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Calculates the 95% Wilson Score confidence interval for Bernoulli trials.
    """
    if n <= 0:
        return (0.0, 0.0)
    p_hat = float(k) / float(n)
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat)) / n + (z**2) / (4.0 * (n**2)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained SAC Policy vs Nominal Bézier")
    parser.add_argument("--checkpoint", type=str, default=str(WS_ROOT / "checkpoints" / "sac_latest.pt"),
                        help="Path to trained SAC checkpoint .pt file")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Stage to evaluate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of test episodes per method")
    parser.add_argument("--horizon", type=int, default=20, help="RL horizon steps")
    parser.add_argument("--substeps", type=int, default=35, help="Simulation substeps")
    parser.add_argument("--show_viewer", action="store_true", help="Render 3D viewer")
    parser.add_argument("--record", action="store_true", help="Record MP4 videos of best, worst, and nominal attempts")
    parser.add_argument("--video_dir", type=str, default=str(WS_ROOT / "videos" / "eval_recordings"),
                        help="Output directory for MP4 video recordings")
    parser.add_argument("--anchor_mode", type=str, default="pinned", choices=["pinned", "soft", "free"],
                        help="Passive base anchor mode ('pinned', 'soft', 'free')")
    parser.add_argument("--friction", type=float, default=80.0,
                        help="Table friction coefficient (e.g. 80.0, 40.0, 15.0, 5.0)")
    parser.add_argument("--tau_start", type=float, default=0.5,
                        help="Initial trajectory progress (0.5 for proximity curriculum, 0.0 for full fold)")
    parser.add_argument("--cloth_checkpoint", type=str, default=None,
                        help="Path to cloth state .pt to initialize simulation from")
    parser.add_argument("--release_at_end", action="store_true", default=False,
                        help="Release active grasp at the end of the episode")
    parser.add_argument("--garment_id", type=str, default="430",
                        help="Garment ID from FoldNet dataset (0-999)")
    return parser.parse_args()


def run_trials(
    env: DynamicFoldingEnv,
    agent: Optional[SACAgent],
    stage: int,
    checkpoint_to_load: Optional[str],
    n_episodes: int,
    is_nominal: bool = False,
    record: bool = False,
    video_dir: Optional[Path] = None,
    anchor_mode: str = "pinned",
    tau_start: float = 0.5,
    friction: float = 80.0,
    run_timestamp: Optional[str] = None,
    release_at_end: bool = False,
) -> Dict[str, Any]:
    """
    Runs evaluation episodes for either nominal baseline (a = 0) or deterministic policy.
    Optionally records MP4 videos for each episode.
    """
    action_dim = env.action_space.shape[0]
    results = {
        "returns": [],
        "final_dists": [],
        "successes": [],
        "alignment_errors": [],
        "contact_ratios": [],
    }

    mode_tag = "nominal" if is_nominal else "sac"
    mode_str = "NOMINAL (a = 0)" if is_nominal else "SAC POLÍTICA (determinista)"
    print(f"\n[*] Evaluando modo: {mode_str} | Etapa {stage} | {n_episodes} episodios...")

    if record and video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)

    temp_video_files: List[Path] = []

    for ep in range(1, n_episodes + 1):
        options = {
            "checkpoint_path": checkpoint_to_load,
            "anchor_mode": anchor_mode,
            "tau_start": tau_start,
            "friction": friction,
            "release_at_end": release_at_end,
        }
        obs, info = env.reset(stage=stage, options=options)
        ep_return = 0.0
        done = False

        temp_video_path = None
        if record and video_dir:
            temp_video_path = video_dir / f"temp_{mode_tag}_ep_{ep}.mp4"
            temp_video_files.append(temp_video_path)
            env.start_recording(str(temp_video_path), fps=30.0)

        while not done:
            if is_nominal or agent is None:
                action = np.zeros(action_dim, dtype=np.float32)
            else:
                action = agent.select_action(obs, deterministic=True)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            obs = next_obs

        if record and video_dir:
            env.stop_recording()

        # Registrar métricas
        is_succ = bool(step_info.get("is_success", False))
        final_d = float(step_info.get("dist_to_target", 0.0))
        align_err = float(step_info.get("alignment_error_m", step_info.get("alignment_error", 0.0)))
        contact_r = float(step_info.get("contact_ratio", 0.0))

        results["returns"].append(ep_return)
        results["final_dists"].append(final_d)
        results["successes"].append(is_succ)
        results["alignment_errors"].append(align_err)
        results["contact_ratios"].append(contact_r)

        print(
            f"    Ep {ep:2d}: Return = {ep_return:7.2f} | "
            f"Dist = {final_d:6.4f} m | "
            f"Exito = {'OK' if is_succ else 'FAIL'} | "
            f"Error Align = {align_err:6.4f} m | "
            f"Contacto = {contact_r*100:5.1f}%"
        )

    # Procesar y preservar únicamente los videos de interés (Mejor, Peor y Nominal) con timestamp en el nombre
    if record and video_dir and temp_video_files:
        brain_dir = Path(r"C:\Users\lam70\.gemini\antigravity\brain\33c99c82-af82-4e4d-a9ee-2933275f4812")

        # Formato seguro para nombres de archivo en Windows (sin dos puntos)
        ts_tag = run_timestamp if run_timestamp else time.strftime("%Y%m%d_%H%M%S")

        if is_nominal:
            final_nom_ts = video_dir / f"stage_{stage}_nominal_baseline_{ts_tag}.mp4"
            final_nom_canonical = video_dir / f"stage_{stage}_nominal_baseline.mp4"

            if temp_video_files[0].exists():
                shutil.copy(str(temp_video_files[0]), str(final_nom_ts))
                shutil.copy(str(temp_video_files[0]), str(final_nom_canonical))
                if brain_dir.exists():
                    shutil.copy(str(final_nom_ts), str(brain_dir / final_nom_ts.name))
                    shutil.copy(str(final_nom_ts), str(brain_dir / final_nom_canonical.name))

            for f in temp_video_files:
                if f.exists():
                    f.unlink()
            print(f"    [VIDEO] Baseline nominal guardado en: {final_nom_ts}")
        else:
            best_idx = int(np.argmin(results["final_dists"]))
            worst_idx = int(np.argmax(results["final_dists"]))

            final_best_ts = video_dir / f"stage_{stage}_sac_best_attempt_{ts_tag}.mp4"
            final_best_canonical = video_dir / f"stage_{stage}_sac_best_attempt.mp4"

            final_worst_ts = video_dir / f"stage_{stage}_sac_worst_attempt_{ts_tag}.mp4"
            final_worst_canonical = video_dir / f"stage_{stage}_sac_worst_attempt.mp4"

            if temp_video_files[best_idx].exists():
                shutil.copy(str(temp_video_files[best_idx]), str(final_best_ts))
                shutil.copy(str(temp_video_files[best_idx]), str(final_best_canonical))
                if brain_dir.exists():
                    shutil.copy(str(final_best_ts), str(brain_dir / final_best_ts.name))
                    shutil.copy(str(final_best_ts), str(brain_dir / final_best_canonical.name))

            if temp_video_files[worst_idx].exists():
                shutil.copy(str(temp_video_files[worst_idx]), str(final_worst_ts))
                shutil.copy(str(temp_video_files[worst_idx]), str(final_worst_canonical))
                if brain_dir.exists():
                    shutil.copy(str(final_worst_ts), str(brain_dir / final_worst_ts.name))
                    shutil.copy(str(final_worst_ts), str(brain_dir / final_worst_canonical.name))

            # Limpiar videos temporales
            for f in temp_video_files:
                if f.exists():
                    f.unlink()

            print(f"    [VIDEO] Mejor intento ({results['final_dists'][best_idx]:.4f} m) guardado en: {final_best_ts}")
            print(f"    [VIDEO] Peor intento  ({results['final_dists'][worst_idx]:.4f} m) guardado en: {final_worst_ts}")

    # Cómputo estadístico
    n = len(results["successes"])
    k = sum(results["successes"])
    sr = float(k) / float(n) if n > 0 else 0.0
    ci_low, ci_high = compute_wilson_ci(k, n)

    summary = {
        "n_episodes": n,
        "success_count": k,
        "success_rate": sr,
        "wilson_ci_95": (ci_low, ci_high),
        "mean_return": float(np.mean(results["returns"])),
        "std_return": float(np.std(results["returns"])),
        "mean_final_dist_m": float(np.mean(results["final_dists"])),
        "std_final_dist_m": float(np.std(results["final_dists"])),
        "mean_alignment_error_m": float(np.mean(results["alignment_errors"])),
        "mean_contact_ratio": float(np.mean(results["contact_ratios"])),
    }
    return summary


def evaluate():
    args = parse_args()

    print("=" * 75)
    print("  EVALUACION COMPARATIVA: BASELINE NOMINAL BEZIER VS. POLITICA SAC")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dispositivo: {device.upper()}")

    report_dir = WS_ROOT / "eval_reports"
    video_dir = Path(args.video_dir) if args.record else None
    report_dir.mkdir(parents=True, exist_ok=True)
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)

    # 1. Instanciacion de Entorno (habilitando camara si --record está activo)
    env = DynamicFoldingEnv(
        garment_id=args.garment_id,
        show_viewer=args.show_viewer,
        enable_camera=args.record,
        rl_horizon_steps=args.horizon,
        substeps_per_action=args.substeps,
        action_clip_m=0.03,
        delta_threshold=0.04,
        release_at_end=args.release_at_end,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Instanciacion y Carga de Politica SAC
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=1.0,
        device=device,
    )
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        print(f"[*] Cargando pesos entrenados desde: {ckpt_path}")
        agent.load(str(ckpt_path))
    else:
        print(f"[!] ADVERTENCIA: Checkpoint no encontrado en {ckpt_path}. Usando pesos sin entrenar.")

    if args.cloth_checkpoint:
        stage_ckpt = args.cloth_checkpoint
    elif args.stage == 2:
        if args.tau_start >= 0.60:
            stage_ckpt = "D:/genesis_ws/checkpoints/Canonicals/pantalon_430_stage2_pose_c_grounded.pt"
        elif args.tau_start >= 0.25:
            stage_ckpt = "D:/genesis_ws/checkpoints/Canonicals/pantalon_430_stage2_pose_b_airborne.pt"
        else:
            stage_ckpt = "D:/genesis_ws/checkpoints/Canonicals/pantalon_430_folded_stage1.pt"
    else:
        curriculum = CurriculumManager(checkpoints_dir="D:/genesis_ws/checkpoints/Canonicals")
        stage_ckpt = curriculum.get_checkpoint_for_stage(args.stage)

    # Si estamos en Etapa 2 y el usuario no especificó anchor_mode, usamos "free"
    anchor_mode = args.anchor_mode
    if args.stage == 2 and anchor_mode == "pinned":
        anchor_mode = "free"

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    display_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        # A. Evaluacion del Baseline Nominal (a = 0)
        nominal_summary = run_trials(
            env=env,
            agent=None,
            stage=args.stage,
            checkpoint_to_load=stage_ckpt,
            n_episodes=args.episodes,
            is_nominal=True,
            record=args.record,
            video_dir=video_dir,
            anchor_mode=anchor_mode,
            tau_start=args.tau_start,
            friction=args.friction,
            run_timestamp=run_timestamp,
            release_at_end=args.release_at_end,
        )

        # B. Evaluacion de la Politica Entrenada SAC
        sac_summary = run_trials(
            env=env,
            agent=agent,
            stage=args.stage,
            checkpoint_to_load=stage_ckpt,
            n_episodes=args.episodes,
            is_nominal=False,
            record=args.record,
            video_dir=video_dir,
            anchor_mode=anchor_mode,
            tau_start=args.tau_start,
            friction=args.friction,
            run_timestamp=run_timestamp,
            release_at_end=args.release_at_end,
        )

        # C. Tabla Comparativa Final
        print("\n" + "=" * 75)
        print("  RESUMEN COMPARATIVO ESTADISTICO (Wilson Score CI 95%)")
        print("=" * 75)
        print(f"{'Metrica':<30} | {'Baseline Nominal':<18} | {'SAC Policy':<18}")
        print("-" * 75)

        nom_sr_str = f"{nominal_summary['success_rate']*100:5.1f}% [{nominal_summary['wilson_ci_95'][0]*100:.1f}%, {nominal_summary['wilson_ci_95'][1]*100:.1f}%]"
        sac_sr_str = f"{sac_summary['success_rate']*100:5.1f}% [{sac_summary['wilson_ci_95'][0]*100:.1f}%, {sac_summary['wilson_ci_95'][1]*100:.1f}%]"
        print(f"{'Tasa de Exito (SR)':<30} | {nom_sr_str:<18} | {sac_sr_str:<18}")

        nom_dist_str = f"{nominal_summary['mean_final_dist_m']*1000:6.1f} mm"
        sac_dist_str = f"{sac_summary['mean_final_dist_m']*1000:6.1f} mm"
        print(f"{'Distancia Media Final':<30} | {nom_dist_str:<18} | {sac_dist_str:<18}")

        nom_align_str = f"{nominal_summary['mean_alignment_error_m']*1000:6.1f} mm"
        sac_align_str = f"{sac_summary['mean_alignment_error_m']*1000:6.1f} mm"
        print(f"{'Error Alineacion Media':<30} | {nom_align_str:<18} | {sac_align_str:<18}")

        nom_cont_str = f"{nominal_summary['mean_contact_ratio']*100:5.1f}%"
        sac_cont_str = f"{sac_summary['mean_contact_ratio']*100:5.1f}%"
        print(f"{'Ratio Contacto Mesa':<30} | {nom_cont_str:<18} | {sac_cont_str:<18}")

        nom_ret_str = f"{nominal_summary['mean_return']:6.2f} +/- {nominal_summary['std_return']:4.2f}"
        sac_ret_str = f"{sac_summary['mean_return']:6.2f} +/- {sac_summary['std_return']:4.2f}"
        print(f"{'Retorno Medio':<30} | {nom_ret_str:<18} | {sac_ret_str:<18}")
        print("=" * 75)

        # Guardar reporte JSON
        report = {
            "stage": args.stage,
            "n_episodes": args.episodes,
            "timestamp": display_timestamp,
            "run_tag": run_timestamp,
            "checkpoint": str(args.checkpoint),
            "nominal_baseline": nominal_summary,
            "sac_policy": sac_summary,
        }
        report_file_ts = report_dir / f"eval_stage_{args.stage}_{run_timestamp}_report.json"
        report_file_canonical = report_dir / f"eval_stage_{args.stage}_report.json"
        with open(report_file_ts, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open(report_file_canonical, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[*] Reporte completo guardado en: {report_file_ts}")
        if args.record and video_dir:
            print(f"[*] Videos exportados con HUD y timestamp en: {video_dir}")

    finally:
        print("[*] Cerrando backend de simulacion...")
        env.close()


if __name__ == "__main__":
    evaluate()
