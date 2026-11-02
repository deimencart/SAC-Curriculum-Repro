"""
measure_corner_fold.py
======================
Script de diagnostico para medir el desplazamiento real de r_corner
durante y despues del Stage 1 del plegado.

Ejecutar con:
    .venv/Scripts/python.exe phase_5_gym_env/cloth_curriculum_rl/envs/measure_corner_fold.py
"""

import numpy as np
import torch

try:
    from cloth_genesis_backend import ClothGenesisBackend
except ImportError:
    from envs.cloth_genesis_backend import ClothGenesisBackend
from bezier_geometry import BezierTrajectory


def measure_r_corner_trajectory():
    backend = ClothGenesisBackend(show_viewer=False, n_envs=1)

    # --- Reset y asentamiento ---
    backend.reset(stage=1, n_settle_steps=40)

    # --- Captura posicion inicial de ambas esquinas ANTES de fijar nada ---
    pos_before = backend.get_particles_pos()
    r_corner_before = pos_before[backend.keypoint_indices["r_corner"]].copy()
    l_corner_before = pos_before[backend.keypoint_indices["l_corner"]].copy()
    print(f"[INIT] l_corner: {l_corner_before}")
    print(f"[INIT] r_corner: {r_corner_before}")
    print(f"[INIT] distancia l<->r: {np.linalg.norm(l_corner_before - r_corner_before):.4f} m")

    # --- Bind Stage 1: fija l_corner, target = posicion de r_corner (levemente alzada) ---
    info = backend.bind_stage_keypoints(stage=1, anchor_mode="free")
    p0 = info["p0"]
    p_target = info["p_target"]
    print(f"\n[STAGE 1] p0 (l_corner): {p0}")
    print(f"[STAGE 1] p_target: {p_target}")

    # --- Trayectoria Bezier ---
    traj = BezierTrajectory(
        p0=np.asarray(p0, dtype=np.float32),
        p_target=np.asarray(p_target, dtype=np.float32),
        h_ratio=info.get("h_ratio", 0.48),
        theta_deg=info.get("theta_deg", 12.0),
        mode=info.get("mode", "cubic"),
    )

    n_fold_steps = info.get("n_fold_steps", 550)
    log_interval = max(1, n_fold_steps // 10)

    print(f"\n[FOLD] Ejecutando {n_fold_steps} pasos de plegado...")
    print(f"{'Step':>6}  {'tau':>5}  {'l_corner xyz':>30}  {'r_corner xyz':>30}  {'r_disp(m)':>10}")
    print("-" * 90)

    for step_idx in range(n_fold_steps):
        tau = (step_idx + 1) / n_fold_steps
        waypoint = traj.get_position(tau)
        backend.step_trajectory(target_pos=waypoint, substeps=35)

        if (step_idx + 1) % log_interval == 0 or step_idx == n_fold_steps - 1:
            pos_now = backend.get_particles_pos()
            l_now = pos_now[backend.keypoint_indices["l_corner"]]
            r_now = pos_now[backend.keypoint_indices["r_corner"]]
            r_disp = float(np.linalg.norm(r_now - r_corner_before))
            print(
                f"{step_idx+1:>6}  {tau:>5.2f}  "
                f"{l_now[0]:>8.4f},{l_now[1]:>8.4f},{l_now[2]:>8.4f}  "
                f"{r_now[0]:>8.4f},{r_now[1]:>8.4f},{r_now[2]:>8.4f}  "
                f"{r_disp:>10.4f}"
            )

    # --- Asentamiento final ---
    print("\n[SETTLE] Liberando y asentando...")
    backend.release_and_settle(n_settle_steps=60)

    pos_final = backend.get_particles_pos()
    l_final = pos_final[backend.keypoint_indices["l_corner"]]
    r_final = pos_final[backend.keypoint_indices["r_corner"]]

    r_total_disp = float(np.linalg.norm(r_final - r_corner_before))
    l_total_disp = float(np.linalg.norm(l_final - l_corner_before))
    align_error = float(np.linalg.norm(l_final - r_final))

    print("\n" + "=" * 60)
    print("[RESULTADO FINAL TRAS PLIEGUE]")
    print(f"  l_corner antes:  {l_corner_before}")
    print(f"  l_corner final:  {l_final}")
    print(f"  r_corner antes:  {r_corner_before}")
    print(f"  r_corner final:  {r_final}")
    print(f"\n  Desplazamiento r_corner: {r_total_disp:.4f} m")
    print(f"  Desplazamiento l_corner: {l_total_disp:.4f} m")
    print(f"\n  Error de alineacion l<->r: {align_error:.4f} m")
    print(f"  Delta XYZ (r_final - r_before): dx={r_final[0]-r_corner_before[0]:.4f}, "
          f"dy={r_final[1]-r_corner_before[1]:.4f}, dz={r_final[2]-r_corner_before[2]:.4f}")

    # Criterio de calidad: el doble pliegue sera correcto si el error de alineacion es pequeno
    if align_error <= 0.040:
        print("\n[OK] Alineacion correcta para el doble pliegue (error <= 4 cm).")
    else:
        print(f"\n[WARN] Alineacion deficiente: {align_error:.4f} m > 4 cm.")
        print("       El doble pliegue movera la malla porque r_corner se desplazo demasiado.")
        print(f"       Solucion sugerida: usar anchor_mode='pinned' o 'soft' en Stage 1.")

    return {
        "r_corner_before": r_corner_before,
        "r_corner_final": r_final,
        "l_corner_final": l_final,
        "r_total_disp_m": r_total_disp,
        "align_error_m": align_error,
    }


if __name__ == "__main__":
    results = measure_r_corner_trajectory()
    print("\n[DICT] Resultados:", results)

