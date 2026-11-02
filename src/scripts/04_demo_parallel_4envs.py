"""
04_demo_parallel_4envs.py
=========================
Demostración de simulación y doblado en paralelo con 4 entornos simultáneos en GPU.
Genera un video panorámico de la cuadrícula 2x2 mostrando las 4 prendas plegándose
con rendimiento y telemetría tensorial en tiempo real.
"""

import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch

WS_ROOT = Path(__file__).resolve().parent.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from envs.dynamic_folding_env import DynamicFoldingEnv
from models.sac_policy import SACAgent


def main():
    parser = argparse.ArgumentParser(description="Demostración de 4 Entornos Paralelos en GPU")
    parser.add_argument("--horizon", type=int, default=20, help="Pasos RL de horizonte")
    parser.add_argument("--substeps", type=int, default=35, help="Subpasos de integración física")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Ejecutar política en modo determinista")
    parser.add_argument("--record", action="store_true", default=True, help="Grabar video MP4")
    parser.add_argument("--output", type=str, default="videos/demo_parallel_4envs.mp4", help="Ruta del video de salida")
    args = parser.parse_args()

    print("=" * 75)
    print("  DEMOSTRACION: 4 ENTORNOS PARALELOS EN GPU (GENESIS XPBD)")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dispositivo: {device.upper()}")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM Inicial: {torch.cuda.memory_allocated() / (1024**2):.1f} MB")

    out_path = WS_ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Creando entorno con n_envs=4 y cámara panorámica HD...")
    env = DynamicFoldingEnv(
        show_viewer=False,
        enable_camera=args.record,
        rl_horizon_steps=args.horizon,
        substeps_per_action=args.substeps,
        n_envs=4,
    )

    print("\n[2/4] Reiniciando los 4 entornos en Etapa 1 (Plegado lateral)...")
    t0 = time.time()
    obs, info = env.reset(stage=1)
    reset_time = time.time() - t0

    print(f"      Observación tensorial: {obs.shape} (4 entornos x 22 dimensiones)")
    print(f"      Partículas totales simuladas en GPU: {env.backend.get_particles_pos().shape}")
    print(f"      Tiempo de reset y asentamiento: {reset_time:.2f} s")

    if args.record:
        print(f"\n[3/4] Iniciando grabación panorámica en: {out_path}...")
        env.start_recording(str(out_path), fps=30.0)

    obs_dim = env.observation_space.shape[0]
    agent = SACAgent(obs_dim=obs_dim, action_dim=3, device=device)
    ckpt_path = WS_ROOT / "checkpoints" / "sac_best_stage1.pt"
    if ckpt_path.exists():
        print(f"[*] Cargando pesos entrenados desde: {ckpt_path.name}")
        agent.load(str(ckpt_path))

    print("\n[4/4] Ejecutando doblado simultáneo a lo largo de 20 pasos de horizonte...")
    t_start_sim = time.time()
    total_steps = 0

    for step_i in range(1, args.horizon + 1):
        # Seleccionamos acciones para los 4 entornos
        action = agent.select_action(obs, deterministic=args.deterministic) # shape: (4, 3)

        next_obs, reward, term, trunc, step_info = env.step(action)
        total_steps += 4
        obs = next_obs

        dists = step_info["dist_to_target"]
        dists_str = ", ".join([f"{d*1000:4.1f}mm" for d in dists])
        print(f"    Paso {step_i:2d}/{args.horizon:2d} | Distancias finales: [{dists_str}] | Recompensas: {reward}")

    sim_time = time.time() - t_start_sim
    fps_rl = total_steps / max(1e-5, sim_time)

    if args.record:
        env.stop_recording()
        print(f"\n[+] Video guardado con éxito en: {out_path}")

    env.close()

    print("\n" + "=" * 75)
    print("  RESULTADOS DE RENDIMIENTO VECTORIZADO")
    print("=" * 75)
    print(f"  - Entornos paralelos activos:  4")
    print(f"  - Muestras RL procesadas:      {total_steps} transiciones en {sim_time:.2f} s")
    print(f"  - Throughput efectivo de RL:   {fps_rl:.1f} muestras/segundo")
    if device == "cuda":
        print(f"  - VRAM máxima reservada:       {torch.cuda.max_memory_reserved() / (1024**2):.1f} MB")
    print("=" * 75)
    print("\n[OK] Demostración completada con éxito.")


if __name__ == "__main__":
    main()
