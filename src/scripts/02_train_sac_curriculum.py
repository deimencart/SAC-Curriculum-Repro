"""
02_train_sac_curriculum.py
==========================
Staged Training Loop for Soft Actor-Critic (SAC) with Adaptive Curriculum Learning.
Coordinates:
1. Environment instantiation (DynamicFoldingEnv with Genesis XPBD backend).
2. Soft Actor-Critic agent (GaussianActor, TwinCritic, learnable temperature alpha).
3. Experience replay buffer with GPU batch sampling.
4. Adaptive curriculum manager: rolling success window, table friction decay, and stage promotion (Stage 1 -> 2 -> 3).
5. Robust metric logging (JSONL/CSV) and automatic checkpoint saving.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml

# Add project root to sys.path
WS_ROOT = Path(__file__).resolve().parent.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from envs.dynamic_folding_env import DynamicFoldingEnv
from models.sac_policy import SACAgent, ReplayBuffer
from callbacks.curriculum_callback import CurriculumManager


def parse_args():
    parser = argparse.ArgumentParser(description="SAC Curriculum Training for Cloth Folding")
    parser.add_argument("--episodes", type=int, default=100, help="Total training episodes")
    parser.add_argument("--horizon", type=int, default=20, help="RL horizon steps per episode")
    parser.add_argument("--substeps", type=int, default=35, help="Simulation substeps per action")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for SAC updates")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Random exploration steps before training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--buffer_capacity", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--save_interval", type=int, default=10, help="Save model checkpoint every N episodes")
    parser.add_argument("--show_viewer", action="store_true", help="Launch interactive 3D viewer (slower)")
    parser.add_argument("--start_stage", type=int, default=1, choices=[1, 2, 3, 4], help="Initial stage to train")
    parser.add_argument("--max_stage", type=int, default=4, choices=[1, 2, 3, 4], help="Maximum stage to train")
    parser.add_argument("--start_level", type=int, default=1, choices=list(range(1, 9)), help="Curriculum level in stage (1 to 8 in Stage 1, 1 to 5 in Stage 2)")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments in GPU (e.g. 1 or 4)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt to resume/warm-start from")
    parser.add_argument("--garment_id", type=str, default="430", help="Garment ID from FoldNet dataset (0-999)")
    parser.add_argument("--dataset_dir", type=str, default="D:/genesis_ws/FoldNet_2D/mesh/trousers", help="Path to FoldNet dataset")
    parser.add_argument("--randomize_garment", action="store_true", help="Enable random sampling across FoldNet garment pool")
    parser.add_argument("--garment_pool", type=str, default=None, help="Comma-separated garment IDs (e.g. '0,1,10,50,100,430,750,999')")
    parser.add_argument("--use_real_scene", action="store_true", default=True, help="Use real lab setup (Campero + Table 3 + UR10e)")
    parser.add_argument("--export_best_trajectory", action="store_true", default=True, help="Export best trajectory in /campero_ur10_base_link frame for physical robot execution")
    return parser.parse_args()


def load_config() -> Dict[str, Any]:
    config_path = WS_ROOT / "configs" / "training_params.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def train():
    args = parse_args()
    config = load_config()

    print("=" * 75)
    print("  ENTRENAMIENTO SAC CON CURRICULUM: PLEGADO DINAMICO DE TELA (GENESIS)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dispositivo: {device.upper()}")
    if device == "cuda":
        print(f"[*] Dispositivo GPU: {torch.cuda.get_device_name(0)}")

    # Carpetas de checkpoints y logs
    ckpt_dir = WS_ROOT / "checkpoints"
    log_dir = WS_ROOT / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_log_path = log_dir / "training_metrics.jsonl"
    csv_log_path = log_dir / "training_metrics.csv"

    # Escribir cabecera CSV si es archivo nuevo
    if not csv_log_path.exists():
        with open(csv_log_path, "w", encoding="utf-8") as f:
            f.write("episode,stage,ep_reward,final_dist_m,is_success,success_rate,friction,critic_loss,actor_loss,alpha,time_sec\n")

    # 1. Instanciacion de Entorno
    print("\n[+] 1. Inicializando entorno de simulacion física...")
    action_clip_m = float(config.get("task_env", {}).get("action_clip_meters", 0.03))
    delta_threshold = float(config.get("task_env", {}).get("delta_threshold", 0.04))

    garment_pool = [x.strip() for x in args.garment_pool.split(",") if x.strip()] if args.garment_pool else None
    env = DynamicFoldingEnv(
        garment_id=args.garment_id,
        dataset_dir=args.dataset_dir,
        randomize_garment=args.randomize_garment,
        garment_pool=garment_pool,
        show_viewer=args.show_viewer,
        rl_horizon_steps=args.horizon,
        substeps_per_action=args.substeps,
        action_clip_m=action_clip_m,
        delta_threshold=delta_threshold,
        n_envs=args.n_envs,
        use_real_scene=args.use_real_scene,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Instanciacion de Agente SAC y Buffer
    print("[+] 2. Inicializando Agente SAC y ReplayBuffer...")
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=1.0,
        lr=args.lr,
        gamma=0.99,
        tau=0.005,
        device=device,
    )

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"[*] Cargando pesos previos (warm-start) desde: {args.checkpoint}")
        agent.load(args.checkpoint)
        if args.warmup_steps == 50:
            args.warmup_steps = 0
            print("[*] Pasos de warmup reducidos a 0 para aprovechar la politica aprendida.")
    elif args.checkpoint:
        print(f"[!] ADVERTENCIA: Checkpoint no encontrado en {args.checkpoint}. Entrenando desde cero.")

    buffer = ReplayBuffer(
        capacity=args.buffer_capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    # 3. Gestor de Curriculo
    curric_cfg = config.get("curriculum", {})
    friction_min = float(curric_cfg.get("friction_min", 5.0))
    friction_max = float(curric_cfg.get("friction_max", 80.0))
    curriculum = CurriculumManager(
        target_success_rate=0.80,
        window_size=12,
        friction_min=friction_min,
        friction_max=friction_max,
        max_stage=args.max_stage,
        start_level=args.start_level,
        checkpoints_dir="D:/genesis_ws/checkpoints/Canonicals",
    )
    curriculum.set_stage(args.start_stage, args.start_level)

    print(f"[*] Parametros de entrenamiento:")
    print(f"    - Entornos en paralelo:  {args.n_envs}")
    print(f"    - Episodios totales:     {args.episodes}")
    print(f"    - Pasos RL por episodio: {args.horizon}")
    print(f"    - Batch size:            {args.batch_size}")
    print(f"    - Pasos Warmup:          {args.warmup_steps}")
    print(f"    - Etapa inicial:         {curriculum.current_stage}")
    print(f"    - Nivel curricular:      {curriculum.current_config['level']} ({curriculum.current_config['desc']})")
    print(f"    - Tau inicial (cercanía):{curriculum.current_config['tau_start']}")
    print(f"    - Sujeción pernera base: {curriculum.current_config['anchor_mode']}")
    print(f"    - Rango friccion mesa:   [{friction_min}, {friction_max}]")

    print("\n" + "-" * 85)
    print(f"{'Ep':>4} | {'Stg':>3} | {'Reward':>8} | {'Dist(cm)':>8} | {'Drift(cm)':>9} | {'Succ':>4} | {'SR(%)':>5} | {'Fric':>5} | {'CritLoss':>8} | {'ActLoss':>7} | {'Alpha':>6} | {'t(s)':>5}")
    print("-" * 85)

    total_steps = 0
    best_stage_return = {1: -float("inf"), 2: -float("inf"), 3: -float("inf"), 4: -float("inf")}

    try:
        for ep in range(1, args.episodes + 1):
            ep_start_time = time.time()
            stage = curriculum.current_stage

            # Opciones de reinicio segun la etapa y el nivel curricular (3 ejes)
            options = curriculum.get_env_options()
            obs, info = env.reset(stage=stage, options=options)
            ep_reward = 0.0
            done = False
            last_losses = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": agent.alpha.item()}
            episode_waypoints = []

            while not done:
                # Politica de exploracion inicial (Warmup) vs SAC
                if total_steps < args.warmup_steps:
                    if args.n_envs > 1:
                        action = np.random.uniform(-1.0, 1.0, size=(args.n_envs, action_dim)).astype(np.float32)
                    else:
                        action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
                else:
                    action = agent.select_action(obs, deterministic=False)

                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated

                if "robot_state" in step_info and step_info["robot_state"]:
                    st = step_info["robot_state"]
                    ee_pos = st["ee_pos"][0] if (args.n_envs > 1 and st["ee_pos"].ndim > 1) else st["ee_pos"]
                    ee_quat = st["ee_quat"][0] if (args.n_envs > 1 and st["ee_quat"].ndim > 1) else st["ee_quat"]
                    joint_pos = st["joint_pos"][0] if (args.n_envs > 1 and st["joint_pos"].ndim > 1) else st["joint_pos"]
                    episode_waypoints.append({
                        "step": len(episode_waypoints),
                        "tau": float(step_info.get("tau", 0.0)),
                        "ee_pos_world": [float(x) for x in ee_pos],
                        "ee_quat_world": [float(x) for x in ee_quat],
                        "ee_pos_robot_base": [float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2] - 0.72)],
                        "joint_pos": [float(q) for q in joint_pos[:6]],
                    })

                if args.n_envs > 1:
                    buffer.add_batch(obs, action, reward, next_obs, np.full(args.n_envs, done))
                    total_steps += args.n_envs
                    ep_reward += float(np.mean(reward))
                else:
                    buffer.push(obs, action, reward, next_obs, done)
                    total_steps += 1
                    ep_reward += reward

                obs = next_obs

                # Paso de optimizacion si hay suficientes datos en el buffer
                if len(buffer) >= args.batch_size and total_steps >= args.warmup_steps:
                    last_losses = agent.update(buffer, batch_size=args.batch_size)

            # Evaluacion curricular del episodio
            if args.n_envs > 1:
                is_success = bool(np.mean(step_info.get("is_success", False)) >= 0.5)
                final_dist = float(np.mean(step_info.get("dist_to_target", 0.0)))
            else:
                is_success = bool(step_info.get("is_success", False))
                final_dist = float(step_info.get("dist_to_target", 0.0))

            curric_res = curriculum.record_episode(stage=stage, episode_return=ep_reward, success=is_success)

            sr = curric_res["success_rate"]
            fric = curric_res["table_friction"]
            ep_time = time.time() - ep_start_time

            # Imprimir linea de progreso
            print(
                f"{ep:4d} | {stage:3d} | {ep_reward:8.2f} | {final_dist*100:8.2f} | "
                f"{float(step_info.get('mean_base_drift', 0.0))*100:9.2f} | "
                f"{'OK' if is_success else 'FAIL':>4} | {sr*100:5.1f} | {fric:5.1f} | "
                f"{last_losses['critic_loss']:8.2f} | {last_losses['actor_loss']:7.2f} | "
                f"{last_losses['alpha']:6.3f} | {ep_time:5.1f}"
            )

            # Registro de metricas en JSONL y CSV
            metric_entry = {
                "episode": ep,
                "stage": stage,
                "ep_reward": ep_reward,
                "final_dist_m": final_dist,
                "alignment_error_m": float(step_info.get("alignment_error_m", step_info.get("alignment_error", 0.0))),
                "fold_dist_xy": float(step_info.get("fold_dist_xy", 0.0)),
                "is_success": is_success,
                "align_ok": bool(step_info.get("align_ok", False)),
                "drift_ok": bool(step_info.get("drift_ok", False)),
                "particles_up_ok": bool(step_info.get("particles_up_ok", False)),
                "span_ok": bool(step_info.get("span_ok", False)),
                "contact_ratio": float(step_info.get("contact_ratio", 0.0)),
                "base_drift": float(step_info.get("base_drift", 0.0)),
                "mean_base_drift": float(step_info.get("mean_base_drift", 0.0)),
                "max_base_drift": float(step_info.get("max_base_drift", 0.0)),
                "r_corner_mean_disp": float(step_info.get("r_corner_mean_disp", 0.0)),
                "r_corner_max_disp": float(step_info.get("r_corner_max_disp", 0.0)),
                "r_corner_drift_dx": float(step_info.get("r_corner_drift_dx", 0.0)),
                "r_corner_drift_dy": float(step_info.get("r_corner_drift_dy", 0.0)),
                "r_corner_drift_dz": float(step_info.get("r_corner_drift_dz", 0.0)),
                "v_factor": float(step_info.get("v_factor", 1.0)),
                "h_lift_boost": float(step_info.get("h_lift_boost", 0.0)),
                "fuse_tripped": bool(step_info.get("fuse_tripped", False)),
                "curriculum_level": int(curric_res.get("level", 1)),
                "anchor_mode": str(curric_res.get("anchor_mode", "free")),
                "tau_start": float(curric_res.get("tau_start", 0.0)),
                "success_rate": sr,
                "friction": fric,
                "critic_loss": last_losses["critic_loss"],
                "actor_loss": last_losses["actor_loss"],
                "alpha": last_losses["alpha"],
                "time_sec": ep_time,
                "total_steps": total_steps,
            }
            with open(jsonl_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metric_entry) + "\n")

            with open(csv_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{ep},{stage},{ep_reward:.4f},{final_dist:.4f},{is_success},"
                    f"{sr:.4f},{fric:.2f},{last_losses['critic_loss']:.4f},"
                    f"{last_losses['actor_loss']:.4f},{last_losses['alpha']:.4f},{ep_time:.2f}\n"
                )

            # Notificar avance de nivel curricular en Stage 1 (sujecion, cercania, friccion)
            if curric_res.get("promoted_level", False):
                print(f"\n{'-'*75}")
                print(f"  [CURRICULUM LEVEL UP] Nivel {curric_res['level']}: {curric_res.get('level_desc', '')}")
                print(f"  Modo Sujecion: {curric_res['anchor_mode']} | Tau Start: {curric_res['tau_start']:.2f} | Friccion: {curric_res['friction']}")
                print(f"{'-'*75}\n")

            # Notificar promocion de etapa si el curriculo lo decidio
            if curric_res.get("should_promote", False):
                print(f"\n{'*'*75}")
                print(f"  [PROMOCION CURRICULAR] Etapa {stage} superada con SR={sr*100:.1f}%.")
                print(f"  Avanzando a Etapa {curric_res['stage']}...")
                print(f"{'*'*75}\n")
                # Guardar checkpoint de superacion de etapa
                stage_ckpt_path = ckpt_dir / f"sac_stage_{stage}_promoted.pt"
                agent.save(str(stage_ckpt_path))

            # Guardado periodico y del mejor modelo (prohibido guardar si se disparo el fusible)
            fuse_was_tripped = bool(step_info.get("fuse_tripped", False))
            if ep_reward > best_stage_return[stage] and not fuse_was_tripped:
                best_stage_return[stage] = ep_reward
                agent.save(str(ckpt_dir / f"sac_best_stage{stage}.pt"))
                if args.export_best_trajectory and len(episode_waypoints) > 0:
                    traj_export_path = ckpt_dir / f"sac_best_stage{stage}_trajectory.json"
                    traj_data = {
                        "stage": int(stage),
                        "episode": int(ep),
                        "reward": float(ep_reward),
                        "final_dist_m": float(final_dist),
                        "n_waypoints": len(episode_waypoints),
                        "frame_id": "/campero_ur10_base_link",
                        "robot_base_pos_world": [0.0, 0.0, 0.72],
                        "table_top_m": float(getattr(env.backend, "table_top", 0.428)),
                        "waypoints": episode_waypoints,
                    }
                    with open(traj_export_path, "w", encoding="utf-8") as f_traj:
                        json.dump(traj_data, f_traj, indent=2)

            if ep % args.save_interval == 0:
                agent.save(str(ckpt_dir / "sac_latest.pt"))

        print("\n" + "=" * 75)
        print("  [FINALIZADO] ENTRENAMIENTO COMPLETADO EXITOSAMENTE.")
        print(f"  Ultimo checkpoint guardado en: {ckpt_dir / 'sac_latest.pt'}")
        print("=" * 75)

    except KeyboardInterrupt:
        print("\n[!] Entrenamiento interrumpido por el usuario. Guardando estado...")
        agent.save(str(ckpt_dir / "sac_interrupted.pt"))
        print(f"[*] Checkpoint de emergencia guardado en: {ckpt_dir / 'sac_interrupted.pt'}")

    finally:
        print("[*] Liberando recursos del simulador...")
        env.close()


if __name__ == "__main__":
    train()
