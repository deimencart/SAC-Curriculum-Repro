"""
record_policy_video.py
======================
Generates MP4 videos of the trained SAC Stage 1 policy executing the complete folding maneuver:
1. Multi-phase grasp sequence: Hover -> Descent -> Firm jaw closure on fabric.
2. Closed-loop folding trajectory executed with closed Robotiq 2F-85 gripper.
3. Clean jaw release and upward arm retraction.
Captures simultaneously from:
- Wide panoramic camera (robot, base, table, whole garment)
- Close-up detail camera (gripper jaws and fabric pinch site)
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Add project root to sys.path
WS_ROOT = Path(__file__).resolve().parent.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from envs.dynamic_folding_env import DynamicFoldingEnv
from models.sac_policy import SACAgent


def main():
    print("=" * 70)
    print("  GRABACION DE VIDEO: SECUENCIA DE AGARRE VERIFICADO Y DOBLADO SAC")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dispositivo: {device.upper()}")

    ckpt_path = WS_ROOT / "checkpoints" / "sac_best_stage1.pt"
    if not ckpt_path.exists():
        print(f"[!] Checkpoint no encontrado en {ckpt_path}. Usando sac_latest.pt")
        ckpt_path = WS_ROOT / "checkpoints" / "sac_latest.pt"

    video_dir = Path("D:/genesis_ws/videos")
    photo_dir = Path("D:/genesis_ws/photos")
    video_dir.mkdir(parents=True, exist_ok=True)
    photo_dir.mkdir(parents=True, exist_ok=True)

    video_path_wide = video_dir / "stage1_policy_real_scene.mp4"
    video_path_close = video_dir / "stage1_policy_real_scene_closeup.mp4"

    # 1. Instanciar el entorno con cámara activada y escena real
    print("[+] 1. Inicializando entorno Genesis con cámaras wide y close-up...")
    env = DynamicFoldingEnv(
        garment_id="430",
        show_viewer=False,
        enable_camera=True,
        rl_horizon_steps=20,
        substeps_per_action=35,
        action_clip_m=0.03,
        delta_threshold=0.04,
        n_envs=1,
        use_real_scene=True,
        release_at_end=True,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Cargar la política entrenada
    print(f"[+] 2. Cargando política entrenada desde: {ckpt_path}")
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_scale=1.0,
        lr=3e-4,
        device=device,
    )
    agent.load(str(ckpt_path))

    # 3. Iniciar la grabación de video ANTES del reset para capturar el agarre completo
    print(f"[+] 3. Iniciando grabaciones de video en:")
    print(f"      Wide:     {video_path_wide}")
    print(f"      Close-up: {video_path_close}")
    if env.backend.cam is not None:
        env.backend.cam.start_recording(save_to_filename=str(video_path_wide), fps=30.0)
    if getattr(env.backend, "cam_closeup", None) is not None:
        env.backend.cam_closeup.start_recording(save_to_filename=str(video_path_close), fps=30.0)

    # 4. Reiniciar el entorno en Stage 1 (sujeción libre, fricción 80)
    # Ejecuta: Hover -> Descenso vertical -> Cierre de pinzas -> Bloqueo PBD
    print("[+] 4. Ejecutando reset y secuencia de agarre verificado...")
    options = {
        "anchor_mode": "free",
        "friction": 80.0,
        "tau_start": 0.0,
    }
    obs, info = env.reset(stage=1, options=options)

    # Capturar fotos con el agarre firme establecido (T=0)
    if env.backend.cam is not None:
        rgb_w, _, _, _ = env.backend.cam.render(rgb=True, depth=False)
        if isinstance(rgb_w, torch.Tensor): rgb_w = rgb_w.cpu().numpy()
        Image.fromarray((rgb_w * 255).astype(np.uint8) if rgb_w.max() <= 1.0 else rgb_w.astype(np.uint8)).save(
            photo_dir / "stage1_step00_approach.png"
        )
    if getattr(env.backend, "cam_closeup", None) is not None:
        rgb_c, _, _, _ = env.backend.cam_closeup.render(rgb=True, depth=False)
        if isinstance(rgb_c, torch.Tensor): rgb_c = rgb_c.cpu().numpy()
        Image.fromarray((rgb_c * 255).astype(np.uint8) if rgb_c.max() <= 1.0 else rgb_c.astype(np.uint8)).save(
            photo_dir / "stage1_step00_grasp_closeup.png"
        )
        print(f"      [Foto] Agarre firme capturado: {photo_dir / 'stage1_step00_grasp_closeup.png'}")

    step_count = 0
    done = False
    ep_reward = 0.0

    print("[+] 5. Ejecutando política determinista en lazo cerrado...")
    while not done:
        action = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        ep_reward += reward
        step_count += 1

        # Capturar fotos a mitad de trayectoria (ápex del arco de doblado)
        if step_count == 10:
            if env.backend.cam is not None:
                rgb_w, _, _, _ = env.backend.cam.render(rgb=True, depth=False)
                if isinstance(rgb_w, torch.Tensor): rgb_w = rgb_w.cpu().numpy()
                Image.fromarray((rgb_w * 255).astype(np.uint8) if rgb_w.max() <= 1.0 else rgb_w.astype(np.uint8)).save(
                    photo_dir / "stage1_step10_folding_apex.png"
                )
            if getattr(env.backend, "cam_closeup", None) is not None:
                rgb_c, _, _, _ = env.backend.cam_closeup.render(rgb=True, depth=False)
                if isinstance(rgb_c, torch.Tensor): rgb_c = rgb_c.cpu().numpy()
                Image.fromarray((rgb_c * 255).astype(np.uint8) if rgb_c.max() <= 1.0 else rgb_c.astype(np.uint8)).save(
                    photo_dir / "stage1_step10_folding_apex_closeup.png"
                )
            print(f"      [Foto] Ápex capturado: {photo_dir / 'stage1_step10_folding_apex.png'}")

        print(f"      Paso {step_count:2d}/20 | Tau: {step_info.get('tau', 0.0):.2f} | Dist: {step_info.get('dist_to_target', 0.0)*100:.2f} cm")

    # Capturar fotos finales del doblez completado, liberado y asentado
    if env.backend.cam is not None:
        rgb_w, _, _, _ = env.backend.cam.render(rgb=True, depth=False)
        if isinstance(rgb_w, torch.Tensor): rgb_w = rgb_w.cpu().numpy()
        Image.fromarray((rgb_w * 255).astype(np.uint8) if rgb_w.max() <= 1.0 else rgb_w.astype(np.uint8)).save(
            photo_dir / "stage1_step20_final_settled.png"
        )
    if getattr(env.backend, "cam_closeup", None) is not None:
        rgb_c, _, _, _ = env.backend.cam_closeup.render(rgb=True, depth=False)
        if isinstance(rgb_c, torch.Tensor): rgb_c = rgb_c.cpu().numpy()
        Image.fromarray((rgb_c * 255).astype(np.uint8) if rgb_c.max() <= 1.0 else rgb_c.astype(np.uint8)).save(
            photo_dir / "stage1_step20_final_settled_closeup.png"
        )
    print(f"      [Foto] Final asentado capturado: {photo_dir / 'stage1_step20_final_settled.png'}")

    # 6. Finalizar y exportar videos
    print("[+] 6. Finalizando y cerrando videos...")
    if env.backend.cam is not None:
        env.backend.cam.stop_recording()
    if getattr(env.backend, "cam_closeup", None) is not None:
        env.backend.cam_closeup.stop_recording()
    env.close()

    final_dist = step_info.get("dist_to_target", 0.0)
    print("\n" + "=" * 70)
    print(f"  [EXITO] GRABACION DE VIDEOS Y FOTOS COMPLETADA.")
    print(f"  Video Wide:      {video_path_wide}")
    print(f"  Video Close-up:  {video_path_close}")
    print(f"  Distancia final: {final_dist*100:.2f} cm")
    print(f"  Retorno total:   {ep_reward:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
