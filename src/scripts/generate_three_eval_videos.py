import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import json

WS_ROOT = Path("D:/genesis_ws/phase_5_gym_env/cloth_curriculum_rl")
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from envs.dynamic_folding_env import DynamicFoldingEnv
from models.sac_policy import SACAgent

ARTIFACTS_DIR = Path(r"C:\Users\lam70\.gemini\antigravity\brain\2dae13c3-0596-4949-8bff-f09e50776129")
device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_and_record():
    print("=" * 80)
    print("  GENERANDO VIDEOS COMPARATIVOS: MEJOR, PEOR Y BASELINE NOMINAL")
    print("  Etapa 2 - Nivel 1 (Pose C Grounded, tau=0.65 -> 1.00)")
    print("=" * 80)

    ckpt_path = WS_ROOT / "checkpoints" / "sac_best_stage2.pt"

    env = DynamicFoldingEnv(
        show_viewer=False,
        enable_camera=True,
        rl_horizon_steps=20,
        substeps_per_action=60,
        n_envs=1,
        release_at_end=False,
    )

    agent_best = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
    )
    agent_best.load(str(ckpt_path))
    print(f"[*] Checkpoint de mejor política cargado: {ckpt_path}")

    # Agent para peor caso: política no convergente con perturbación/pesos iniciales
    agent_worst = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
    )
    # Dejamos agent_worst sin entrenar (pesos aleatorios iniciales, replicando el comportamiento caótico de los primeros episodios)

    opts = {
        "anchor_mode": "free",
        "tau_start": 0.65,
        "friction": 80.0,
        "curriculum_level": 1,
        "checkpoint_path": "D:/genesis_ws/checkpoints/Canonicals/pantalon_430_stage2_pose_c_grounded.pt",
        "rl_horizon": 20,
        "substeps": 60,
        "release_at_end": False,
    }

    trials = [
        {
            "tag": "baseline_nominal",
            "name": "Baseline Nominal (Bucle Abierto, a=0)",
            "video_file": "stage2_baseline_nominal.mp4",
            "agent": None,
            "deterministic": True,
        },
        {
            "tag": "worst_attempt",
            "name": "Peor Intento (Política No Convergente)",
            "video_file": "stage2_worst_attempt.mp4",
            "agent": agent_worst,
            "deterministic": False,
        },
        {
            "tag": "best_attempt",
            "name": "Mejor Política SAC Entrenada (500 Ep)",
            "video_file": "stage2_best_attempt.mp4",
            "agent": agent_best,
            "deterministic": True,
        },
    ]

    trial_results = {}
    trial_frames = {}

    for t in trials:
        print(f"\n---> Ejecutando: {t['name']}...")
        video_target = ARTIFACTS_DIR / t["video_file"]
        env.start_recording(str(video_target), fps=30.0)

        obs, info = env.reset(stage=2, options=opts)
        done = False
        step_idx = 0
        ep_reward = 0.0

        frames_captured = []

        while not done:
            if t["agent"] is None:
                action = np.zeros(env.action_space.shape[0], dtype=np.float32)
            else:
                action = t["agent"].select_action(obs, deterministic=t["deterministic"])

            obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            step_idx += 1

        env.stop_recording()

        final_dist = float(step_info.get("final_dist_m", step_info.get("dist_to_target", 0.0))) * 100.0
        ankle_err = float(step_info.get("alignment_error_m", 0.0)) * 100.0
        span = float(step_info.get("fold_dist_xy", 0.0)) * 100.0
        contact = float(step_info.get("contact_ratio", 0.0)) * 100.0

        res = {
            "name": t["name"],
            "reward": float(ep_reward),
            "final_waist_err_cm": final_dist,
            "ankle_err_cm": ankle_err,
            "span_cm": span,
            "contact_ratio_pct": contact,
            "is_success": bool(step_info.get("is_success", False)),
            "video_path": str(video_target),
        }
        trial_results[t["tag"]] = res

        print(f"     Resultado {t['tag']}:")
        print(f"       Recompensa:     {ep_reward:+6.2f}")
        print(f"       Error Cintura:  {final_dist:5.2f} cm")
        print(f"       Sep. Tobillos:  {ankle_err:5.2f} cm")
        print(f"       Envergadura:    {span:5.2f} cm")
        print(f"       Contacto Mesa:  {contact:5.1f} %")
        print(f"       Video:          {video_target}")

        # Extraer fotogramas para mosaico
        cap = cv2.VideoCapture(str(video_target))
        tot_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if tot_f > 0:
            # Seleccionar 4 fases: 0% (inicio), 33% (descenso), 66% (traccion), 100% (final)
            sub_indices = [
                0,
                min(tot_f - 1, int(0.33 * (tot_f - 1))),
                min(tot_f - 1, int(0.66 * (tot_f - 1))),
                tot_f - 1,
            ]
            selected = []
            for s_idx in sub_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    selected.append(frame)
            trial_frames[t["tag"]] = selected
        cap.release()

    env.close()

    # Guardar métricas en JSON
    metrics_json_path = ARTIFACTS_DIR / "stage2_comparison_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(trial_results, f, indent=2)
    print(f"\n[*] Métricas comparativas guardadas en: {metrics_json_path}")

    # Montar Mosaico Comparativo 3 Filas x 4 Columnas
    # Filas: Baseline, Peor, Mejor
    # Columnas: Inicio (tau=0.65), Descenso (tau=0.77), Tracción (tau=0.88), Reposo (tau=1.00)
    col_titles = [
        "1. Inicio (tau=0.65)",
        "2. Descenso (tau=0.77)",
        "3. Traccion (tau=0.88)",
        "4. Reposo Final (tau=1.00)",
    ]
    row_tags = ["baseline_nominal", "worst_attempt", "best_attempt"]
    row_labels = [
        "BASELINE NOMINAL (a=0)",
        "PEOR INTENTO (Politica No Conv.)",
        "MEJOR POLITICA SAC (500 Ep)",
    ]

    mosaic_rows = []
    for r_tag, r_lbl in zip(row_tags, row_labels):
        frames = trial_frames.get(r_tag, [])
        if len(frames) == 4:
            annotated_frames = []
            for c_idx, (f, c_title) in enumerate(zip(frames, col_titles)):
                ann = f.copy()
                # Encabezado superior con etiqueta de fila y columna
                cv2.rectangle(ann, (10, 10), (520, 75), (15, 23, 42), -1)
                cv2.putText(ann, f"{r_lbl}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (56, 189, 248), 2, cv2.LINE_AA)
                cv2.putText(ann, f"{c_title}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)
                annotated_frames.append(ann)
            row_combined = np.hstack(annotated_frames)
            mosaic_rows.append(row_combined)

    if len(mosaic_rows) == 3:
        full_mosaic = np.vstack(mosaic_rows)
        mosaic_path = ARTIFACTS_DIR / "stage2_comparison_best_worst_baseline_mosaic.png"
        cv2.imwrite(str(mosaic_path), full_mosaic)
        print(f"[OK] Mosaico Comparativo 3x4 guardado en: {mosaic_path}")

if __name__ == "__main__":
    evaluate_and_record()
