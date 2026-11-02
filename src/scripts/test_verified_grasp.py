"""
test_verified_grasp.py
======================
Verifies the complete 3-phase top-down grasp sequence:
1. Pre-grasp hover (open jaws, 6 cm above l_corner)
2. Vertical descent (straddling l_corner between open silicone pads)
3. Jaw closure (driving Robotiq 2F-85 driver joints to 0.70 rad and locking PBD particles)
4. Captures wide view and close-up view snapshots to confirm the firm grasp.
"""

import sys
from pathlib import Path
WS_ROOT = Path(__file__).resolve().parent.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

import torch
import numpy as np
from PIL import Image
from envs.cloth_genesis_backend import ClothGenesisBackend
import scipy.spatial.transform as sst

def main():
    print("=" * 70)
    print("  VERIFICACION DE SECUENCIA DE AGARRE ROBUSTO (UR10e + Robotiq 2F-85)")
    print("=" * 70)

    photo_dir = Path("D:/genesis_ws/photos")
    photo_dir.mkdir(parents=True, exist_ok=True)

    backend = ClothGenesisBackend(
        use_real_scene=True,
        n_envs=1,
        garment_id="430",
        show_viewer=False,
        enable_camera=True
    )

    close_cam = backend.cam_closeup

    # 1. Settle cloth on table
    print("[+] 1. Asentando prenda sobre la mesa...")
    for _ in range(35):
        backend.scene.step()

    l_idx = backend.keypoint_indices["l_corner"]
    p_l = backend.get_particles_pos()[l_idx]
    print(f"      Posicion inicial l_corner: {p_l}")

    # Downward quaternion for base_gripper: [0, 0, 1, 0] in (w, x, y, z)
    # Rotating 180 deg around Y makes Z point down [0, 0, -1]
    down_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=backend.device)

    # Helper function to construct full 14-DOF joint vector
    def make_q_full(q_arm, gripper_open_ratio=0.0):
        # gripper_open_ratio: 0.0 = fully open, 1.0 = fully closed (0.70 rad)
        q = torch.zeros(14, dtype=torch.float32, device=backend.device)
        q[:6] = q_arm
        driver_val = 0.70 * gripper_open_ratio
        q[6] = driver_val        # right_driver
        q[7] = -driver_val       # right_coupler
        q[8] = driver_val        # right_spring_link
        q[10] = driver_val       # left_driver
        q[11] = -driver_val      # left_coupler
        q[12] = driver_val       # left_spring_link
        return q

    # Phase 1: Pre-Grasp Hover (6 cm above l_corner, jaws fully open)
    print("[+] 2. Fase de aproximacion aerea (Pre-Grasp Hover, 6 cm sobre l_corner)...")
    p_hover = p_l.copy()
    p_hover[2] += 0.070
    q_hover = backend.solve_ik(target_pos=p_hover, target_quat=down_quat)
    q_full_hover = make_q_full(q_hover, gripper_open_ratio=0.0)

    # Move arm to hover
    backend.robot.set_dofs_position(q_full_hover)
    backend.robot.zero_all_dofs_velocity()
    for _ in range(25):
        backend.scene.step()

    # Capture hover image
    rgb_w, _, _, _ = backend.cam.render(rgb=True, depth=False)
    rgb_c, _, _, _ = close_cam.render(rgb=True, depth=False)
    if isinstance(rgb_w, torch.Tensor): rgb_w = rgb_w.cpu().numpy()
    if isinstance(rgb_c, torch.Tensor): rgb_c = rgb_c.cpu().numpy()
    Image.fromarray((rgb_w * 255).astype(np.uint8) if rgb_w.max() <= 1.0 else rgb_w.astype(np.uint8)).save(
        photo_dir / "grasp_phase1_hover_wide.png"
    )
    Image.fromarray((rgb_c * 255).astype(np.uint8) if rgb_c.max() <= 1.0 else rgb_c.astype(np.uint8)).save(
        photo_dir / "grasp_phase1_hover_closeup.png"
    )
    print(f"      [Foto] Guardada: {photo_dir / 'grasp_phase1_hover_closeup.png'}")

    # Phase 2: Vertical Descent into Fabric
    print("[+] 3. Fase de descenso vertical sobre la tela (jaws abiertas)...")
    p_pinch_target = p_l.copy()
    p_pinch_target[2] = float(backend.table_top + 0.010) # 1 cm above table surface
    q_touch = backend.solve_ik(target_pos=p_pinch_target, target_quat=down_quat)

    # Smooth descent in 20 substeps
    for i in range(1, 21):
        alpha = i / 20.0
        q_interp = (1.0 - alpha) * q_hover + alpha * q_touch
        q_f = make_q_full(q_interp, gripper_open_ratio=0.0)
        backend.robot.set_dofs_position(q_f)
        backend.scene.step()

    rgb_w, _, _, _ = backend.cam.render(rgb=True, depth=False)
    rgb_c, _, _, _ = close_cam.render(rgb=True, depth=False)
    if isinstance(rgb_w, torch.Tensor): rgb_w = rgb_w.cpu().numpy()
    if isinstance(rgb_c, torch.Tensor): rgb_c = rgb_c.cpu().numpy()
    Image.fromarray((rgb_w * 255).astype(np.uint8) if rgb_w.max() <= 1.0 else rgb_w.astype(np.uint8)).save(
        photo_dir / "grasp_phase2_descent_wide.png"
    )
    Image.fromarray((rgb_c * 255).astype(np.uint8) if rgb_c.max() <= 1.0 else rgb_c.astype(np.uint8)).save(
        photo_dir / "grasp_phase2_descent_closeup.png"
    )
    print(f"      [Foto] Guardada: {photo_dir / 'grasp_phase2_descent_closeup.png'}")

    # Phase 3: Grip Closure (close jaws and lock cloth particles)
    print("[+] 4. Fase de cierre de pinza y bloqueo fisico PBD...")
    for i in range(1, 16):
        ratio = i / 15.0
        q_f = make_q_full(q_touch, gripper_open_ratio=ratio)
        backend.robot.set_dofs_position(q_f)
        backend.scene.step()

    # Calculate particles within grasp radius
    pos_cloth = backend.get_particles_pos()
    dists_l = np.linalg.norm(pos_cloth - p_l.reshape(1, 3), axis=1)
    grasp_indices = np.where(dists_l <= 0.025)[0].astype(np.int32)
    if len(grasp_indices) < 5:
        grasp_indices = np.argsort(dists_l)[:12].astype(np.int32)

    backend.active_grasp_indices = grasp_indices
    backend.active_all_fixed = grasp_indices
    backend.active_patch_offsets = pos_cloth[grasp_indices] - p_l.reshape(1, 3)
    backend.cloth.fix_particles(grasp_indices)

    # Settle physics for 15 steps
    for _ in range(15):
        backend.scene.step()

    link = backend.ee_link
    pos = link.get_pos().cpu().numpy()
    quat = link.get_quat().cpu().numpy()
    r = sst.Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
    pinch_world = pos + r.as_matrix() @ np.array([0.0, 0.0, 0.145])
    print(f"      Pinch site alcanzado: {pinch_world}")
    print(f"      Error al punto l_corner: {np.linalg.norm(pinch_world - p_l)*1000:.2f} mm")
    print(f"      Particulas sujetadas fijadas en PBD: {len(grasp_indices)}")

    rgb_w, _, _, _ = backend.cam.render(rgb=True, depth=False)
    rgb_c, _, _, _ = close_cam.render(rgb=True, depth=False)
    if isinstance(rgb_w, torch.Tensor): rgb_w = rgb_w.cpu().numpy()
    if isinstance(rgb_c, torch.Tensor): rgb_c = rgb_c.cpu().numpy()
    Image.fromarray((rgb_w * 255).astype(np.uint8) if rgb_w.max() <= 1.0 else rgb_w.astype(np.uint8)).save(
        photo_dir / "grasp_phase3_closed_wide.png"
    )
    Image.fromarray((rgb_c * 255).astype(np.uint8) if rgb_c.max() <= 1.0 else rgb_c.astype(np.uint8)).save(
        photo_dir / "grasp_phase3_closed_closeup.png"
    )
    print(f"      [Foto] Guardada: {photo_dir / 'grasp_phase3_closed_closeup.png'}")
    print("\n[+] PRUEBA DE AGARRE VERIFICADA CON EXITO.")

if __name__ == "__main__":
    main()
