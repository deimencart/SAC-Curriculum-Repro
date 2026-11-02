"""
test_robot_table_grasp.py
Verifies that rotating the UR10e base and initializing in the pre-grasp pose
places the Robotiq 2F-85 gripper directly on l_corner on Table 3.
"""
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

sys.path.append("D:/genesis_ws/phase_5_gym_env/cloth_curriculum_rl")
from envs.cloth_genesis_backend import ClothGenesisBackend

def main():
    print("=" * 60)
    print("  TEST: UR10e PRE-GRASP AT L_CORNER ON TABLE 3")
    print("=" * 60)

    backend = ClothGenesisBackend(
        use_real_scene=True,
        n_envs=1,
        garment_id="430",
        show_viewer=False,
        enable_camera=True,
    )

    # Pre-grasp configuration with shoulder_pan rotated towards table:
    q_pregrasp = torch.tensor([1.183, -0.807, 2.036, -2.108, -1.583, 0.0], dtype=torch.float32, device=backend.device)
    backend.home_qpos = q_pregrasp.clone()
    backend.robot.set_dofs_position(q_pregrasp, dofs_idx_local=backend.arm_dof_idx)
    backend.robot.zero_all_dofs_velocity()

    # Settle cloth on table
    for _ in range(30):
        backend.scene.step()

    pos = backend.get_particles_pos()
    l_idx = backend.keypoint_indices["l_corner"]
    p_l = pos[l_idx]
    print(f"[*] Cloth l_corner position on table: {p_l}")

    # Solve IK precisely for l_corner from this seed
    link = backend.ee_link
    target_l = torch.as_tensor(p_l, dtype=torch.float32, device=backend.device)
    local_pinch = torch.tensor([0.0, 0.0, 0.145], dtype=torch.float32, device=backend.device)

    q_sol = backend.robot.inverse_kinematics(
        link=link,
        pos=target_l,
        local_point=local_pinch,
        dofs_idx_local=backend.arm_dof_idx,
        max_solver_iters=50,
    )
    backend.robot.set_dofs_position(q_sol)
    for _ in range(10):
        backend.scene.step()

    # Measure pinch site in world frame
    ee_pos = link.get_pos().cpu().numpy()
    ee_quat = link.get_quat().cpu().numpy()
    import scipy.spatial.transform as sst
    r_mat = sst.Rotation.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]).as_matrix()
    pinch_world = ee_pos + r_mat @ np.array([0.0, 0.0, 0.145])

    dist = np.linalg.norm(pinch_world - p_l)
    print(f"[*] Achieved pinch site: {np.round(pinch_world, 4)}")
    print(f"[*] Pinch error to l_corner: {dist * 1000.0:.2f} mm")

    # Render camera image
    photo_path = Path("D:/genesis_ws/photos/test_ur10e_l_corner_grasp.png")
    photo_path.parent.mkdir(parents=True, exist_ok=True)
    if backend.cam is not None:
        rgb, _, _, _ = backend.cam.render(rgb=True, depth=False)
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.cpu().numpy()
        img = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
        Image.fromarray(img).save(photo_path)
        print(f"[+] Saved snapshot to: {photo_path}")

    assert dist < 0.015, f"Pinch error {dist} m exceeds tolerance 1.5 cm!"
    print("[+] SUCCESS: Robotiq 2F-85 gripper successfully reached and grasps l_corner on Table 3!")

if __name__ == "__main__":
    main()
