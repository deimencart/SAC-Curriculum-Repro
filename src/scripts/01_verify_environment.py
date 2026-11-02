"""
01_verify_environment.py
========================
Verification of Gymnasium Environment, Genesis Physics, and SAC Architecture.
Performs an end-to-end sanity check:
1. Environment instantiation & nominal trajectory evaluation (a = 0).
2. Stochastic action sampling via SAC GaussianActor (a ~ pi(s)).
3. ReplayBuffer data population.
4. SAC update step (Critic, Actor, Alpha loss convergence without NaNs).
5. CurriculumManager metric recording and adaptation.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to sys.path
WS_ROOT = Path(__file__).resolve().parent.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from envs.dynamic_folding_env import DynamicFoldingEnv
from models.sac_policy import SACAgent, ReplayBuffer
from callbacks.curriculum_callback import CurriculumManager


import argparse


def run_verification(stage: int = 1):
    print("=" * 70)
    print(f"  VERIFICACION INTEGRAL: ENTORNO GENESIS + AGENTE SAC (ETAPA {stage})")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dispositivo de computo detectado: {device.upper()}")
    if device == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------
    # 1. Instanciacion de Entorno, Politica, Buffer y Curriculum
    # -------------------------------------------------------------
    print("\n[+] 1. Instanciando modulos...")
    env = DynamicFoldingEnv(
        show_viewer=False,
        rl_horizon_steps=20 if stage == 2 else 10,
        substeps_per_action=60 if stage == 2 else 35,
        action_clip_m=0.03,
        delta_threshold=0.04,
    )
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"    - Espacio de observacion: Box({obs_dim}) [7 keypoints x 3D + tau + drift metrics]")
    print(f"    - Espacio de accion:      Box({action_dim}) [dx, dy, dz, d_hlift, dv]")

    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=1.0,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        device=device,
    )
    buffer = ReplayBuffer(capacity=1000, obs_dim=obs_dim, action_dim=action_dim, device=device)
    curriculum = CurriculumManager(window_size=5, friction_max=80.0, friction_min=5.0)
    curriculum.set_stage(stage, start_level=1)
    print(f"    - Curriculum configurado para Etapa {stage}, Nivel 1: {curriculum.current_config['desc']}")
    print("    - Agente SAC y ReplayBuffer inicializados correctamente.")

    try:
        # -------------------------------------------------------------
        # 2. Episodio 1: Acciones Nominales (a = 0)
        # -------------------------------------------------------------
        print(f"\n[+] 2. Ejecutando Episodio 1: Trayectoria Nominal Pura (a = 0) en Etapa {stage}...")
        options = curriculum.get_env_options()
        obs, info = env.reset(stage=stage, options=options)
        assert obs.shape == (obs_dim,), f"Dimension de observacion esperada {obs_dim}, recibida {obs.shape}"
        assert not np.isnan(obs).any(), "NaN detectado en observacion inicial."

        ep1_reward = 0.0
        done = False
        step = 0
        while not done:
            nominal_action = np.zeros(action_dim, dtype=np.float32)
            next_obs, reward, terminated, truncated, step_info = env.step(nominal_action)
            done = terminated or truncated

            assert not np.isnan(reward), f"NaN detectado en recompensa en paso {step}"
            assert not np.isnan(next_obs).any(), f"NaN detectado en observacion en paso {step}"

            buffer.push(obs, nominal_action, reward, next_obs, done)
            ep1_reward += reward
            obs = next_obs
            step += 1

        print(f"    - Pasos completados:  {step}")
        print(f"    - Recompensa total:   {ep1_reward:.3f}")
        print(f"    - Distancia final:    {step_info['dist_to_target']:.4f} m (umbral: {env.delta_threshold:.4f} m)")
        print(f"    - Exito nominal:      {step_info.get('is_success', False)}")
        if "alignment_error_m" in step_info:
            print(f"    - Error alineacion:   {step_info['alignment_error_m']:.4f} m")

        # Registro en Curriculum
        curric_res1 = curriculum.record_episode(
            stage=stage, episode_return=ep1_reward, success=step_info.get("is_success", False)
        )
        print(f"    - Curriculum status:  SR={curric_res1['success_rate']*100:.1f}%, Friccion={curric_res1['friction']:.1f}")

        # -------------------------------------------------------------
        # 3. Episodio 2: Acciones Estocasticas SAC (Exploracion)
        # -------------------------------------------------------------
        print(f"\n[+] 3. Ejecutando Episodio 2: Acciones Estocasticas del Actor SAC en Etapa {stage}...")
        options = curriculum.get_env_options()
        obs, info = env.reset(stage=stage, options=options)
        ep2_reward = 0.0
        done = False
        step = 0
        while not done:
            action = agent.select_action(obs, deterministic=False)
            assert action.shape == (action_dim,), f"Forma de accion invalida: {action.shape}"
            assert np.all(np.abs(action) <= 1.0 + 1e-5), f"Accion fuera de limites [-1.0, 1.0]: {action}"

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, done)
            ep2_reward += reward
            obs = next_obs
            step += 1

        print(f"    - Pasos completados:  {step}")
        print(f"    - Recompensa total:   {ep2_reward:.3f}")
        print(f"    - Distancia final:    {step_info['dist_to_target']:.4f} m")
        print(f"    - Buffer ocupacion:   {len(buffer)} transiciones")

        # -------------------------------------------------------------
        # 4. Actualizacion de Gradientes SAC (Clipped Double-Q + Policy)
        # -------------------------------------------------------------
        print("\n[+] 4. Verificando paso de optimizacion SAC (Backpropagation)...")
        batch_size = min(len(buffer), 16)
        losses = agent.update(buffer, batch_size=batch_size)

        for k, v in losses.items():
            assert not np.isnan(v), f"NaN detectado en metrica de perdida {k}: {v}"
            assert not np.isinf(v), f"Inf detectado en metrica de perdida {k}: {v}"
            print(f"    - {k:15s}: {v:.6f}")

        # -------------------------------------------------------------
        # 5. Verificacion de Checkpoint Save & Load
        # -------------------------------------------------------------
        print("\n[+] 5. Verificando persistencia de pesos (Save / Load)...")
        test_ckpt_path = WS_ROOT / "checkpoints" / "test_sac_verify.pt"
        test_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(test_ckpt_path))
        assert test_ckpt_path.exists(), f"No se encontro el archivo guardado en {test_ckpt_path}"
        agent.load(str(test_ckpt_path))
        test_ckpt_path.unlink()  # Limpieza de archivo temporal
        print("    - Checkpoint guardado, restaurado y verificado correctamente.")

        print("\n" + "=" * 70)
        print("  [EXITO] TODAS LAS PRUEBAS DE INTEGRACION COMPLETADAS SATISFACTORIAMENTE.")
        print("=" * 70)

    finally:
        print("[*] Cerrando backend de simulacion...")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    run_verification(stage=args.stage)
