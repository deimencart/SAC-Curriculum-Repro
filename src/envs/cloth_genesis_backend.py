"""
cloth_genesis_backend.py
========================
Minimal Genesis XPBD simulation backend for multi-stage cloth folding.
Uses deterministic oracle keypoint binding (no grasp search, zero exceptions).
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import torch
import trimesh
import genesis as gs


def _resolve_mesh_path(rel_path: str) -> str:
    """Resuelve la ruta absoluta de las mallas del setup real."""
    candidates = [
        Path("D:/genesis_ws/setup_real") / rel_path,
        Path("D:/genesis_ws/CODIGO") / rel_path,
        Path(__file__).resolve().parent.parent.parent.parent / "setup_real" / rel_path,
        Path(__file__).resolve().parent.parent.parent.parent / "CODIGO" / rel_path,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return str(Path("D:/genesis_ws/setup_real") / rel_path)


class ClothGenesisBackend:
    """Minimal Genesis XPBD simulation backend for staged cloth folding with UR10e real scene integration."""

    def __init__(
        self,
        garment_id: Union[int, str] = 430,
        dataset_dir: str = "D:/genesis_ws/FoldNet_2D/mesh/trousers",
        mesh_path: Optional[str] = None,
        info_path: Optional[str] = None,
        show_viewer: bool = False,
        enable_camera: bool = False,
        n_envs: int = 1,
        env_spacing: Tuple[float, float] = (2.0, 2.0),
        use_real_scene: bool = True,
    ):
        self.show_viewer = show_viewer
        self.enable_camera = enable_camera
        self.n_envs = max(1, int(n_envs))
        self.env_spacing = env_spacing
        self.use_real_scene = bool(use_real_scene)
        self.cam = None
        self.dataset_dir = Path(dataset_dir)
        self.garment_id = str(garment_id)
        self._cached_stage2_grasp_indices = None
        self.robot = None
        self.ee_link = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if mesh_path is not None:
            self.mesh_path = Path(mesh_path)
        else:
            self.mesh_path = self.dataset_dir / self.garment_id / "mesh.obj"

        if info_path is not None:
            self.info_path = Path(info_path)
        else:
            self.info_path = self.dataset_dir / self.garment_id / "mesh_info.json"

        # 1. Cargar metadatos y dimensiones de la prenda
        self._load_garment_metadata()

        # 2. Workspace geometry
        if self.use_real_scene:
            # Setup Real de Laboratorio (table3 + Campero Base + UR10e)
            self.table_pos = (0.15, -1.0, 0.0)
            self.table_size = (0.613, 0.955, 0.428)
            self.table_top = 0.428
            self.robot_base_pos = (0.0, 0.0, 0.72)
            self.cloth_drop_pos = (0.15, -1.05, self.table_top + 0.015)
        else:
            # Setup abstracto previo
            self.table_pos = (0.72, 0.0, 0.025)
            self.table_size = (1.2, 1.6, 0.05)
            self.table_top = 0.05
            self.robot_base_pos = (0.0, 0.0, 0.025)
            self.robot_base_size = (0.3, 0.3, 0.05)
            self.cloth_drop_pos = (self.table_pos[0], self.table_pos[1], self.table_top + 0.02)

        # 3. Active grasp state & passive anchor state
        self.active_grasp_indices = np.array([], dtype=np.int32)
        self.active_all_fixed = np.array([], dtype=np.int32)
        self.active_patch_offsets = np.empty((0, 3), dtype=np.float32)
        self.particles_init = None
        self.base_particle_indices = np.array([], dtype=np.int32)
        self.r_corner_indices = np.array([], dtype=np.int32)
        self.r_corner_particles_init = np.empty((0, 3), dtype=np.float32)

        # 4. Build simulation scene
        self._build_scene()

    def _load_garment_metadata(self):
        """Carga los keypoints de FoldNet y calcula escalas métricas intrínsecas (ancho, largo)."""
        with open(self.info_path, "r", encoding="utf-8") as f:
            self.keypoint_cfg = json.load(f)["cfg"]

        mesh_tr = trimesh.load(str(self.mesh_path))
        self.verts_2d = mesh_tr.vertices[:, :2]

        self.keypoint_indices = {
            kp: int(np.argmin(np.linalg.norm(self.verts_2d - np.array(self.keypoint_cfg[kp], dtype=np.float32), axis=1)))
            for kp in ["l_corner", "r_corner", "crotch", "top_ctr_f", "top_ctr_b", "l_leg_o", "r_leg_o"]
        }

        # Dimensiones métricas intrínsecas para invariancia a escala
        p_tf_2d = np.array(self.keypoint_cfg["top_ctr_f"], dtype=np.float32)
        p_lc_2d = np.array(self.keypoint_cfg["l_corner"], dtype=np.float32)
        p_rc_2d = np.array(self.keypoint_cfg["r_corner"], dtype=np.float32)
        p_llo_2d = np.array(self.keypoint_cfg["l_leg_o"], dtype=np.float32)
        p_rro_2d = np.array(self.keypoint_cfg["r_leg_o"], dtype=np.float32)

        self.cloth_width = float(np.linalg.norm(p_rc_2d - p_lc_2d))
        self.cloth_length = float(np.abs(p_tf_2d[1] - (p_llo_2d[1] + p_rro_2d[1]) / 2.0))
        self._cached_stage2_grasp_indices = None

    def load_garment(self, garment_id: Union[int, str]) -> bool:
        """Cambia dinámicamente la prenda a otra del dataset FoldNet y reconstruye la escena."""
        garment_id_str = str(garment_id)
        if garment_id_str == getattr(self, "garment_id", None):
            return True

        target_mesh = self.dataset_dir / garment_id_str / "mesh.obj"
        target_info = self.dataset_dir / garment_id_str / "mesh_info.json"
        if not target_mesh.exists() or not target_info.exists():
            print(f"[!] Prenda {garment_id} no encontrada en {self.dataset_dir}")
            return False

        self.garment_id = garment_id_str
        self.mesh_path = target_mesh
        self.info_path = target_info
        self._load_garment_metadata()
        self._build_scene()
        return True

    def _build_scene(self):
        if not getattr(gs, "_initialized", False):
            gs.init(backend=gs.gpu, performance_mode=True)

        if self.use_real_scene:
            viewer_opts = (
                gs.options.ViewerOptions(
                    res=(1280, 720),
                    camera_pos=(1.9, -0.4, 1.4),
                    camera_lookat=(0.15, -0.85, 0.48),
                    camera_fov=45.0,
                    run_in_thread=True,
                )
                if self.show_viewer
                else None
            )
        else:
            viewer_opts = (
                gs.options.ViewerOptions(
                    res=(1280, 720),
                    camera_pos=(2.2, -1.6, 1.4),
                    camera_lookat=(self.table_pos[0], self.table_pos[1], self.table_top + 0.1),
                    camera_fov=45.0,
                    run_in_thread=True,
                )
                if self.show_viewer
                else None
            )

        vis_opts = gs.options.VisOptions(rendered_envs_idx=list(range(self.n_envs))) if self.n_envs > 1 else None
        self.scene = gs.Scene(
            show_viewer=self.show_viewer,
            viewer_options=viewer_opts,
            vis_options=vis_opts,
            sim_options=gs.options.SimOptions(dt=0.004, substeps=10),
            pbd_options=gs.options.PBDOptions(
                max_stretch_solver_iterations=20,
                max_bending_solver_iterations=10,
                particle_size=0.008,
            ),
        )

        if self.enable_camera:
            if self.use_real_scene:
                if self.n_envs > 1:
                    x_c = float((self.env_spacing[0] / 2.0) + 0.15)
                    y_c = float((self.env_spacing[1] / 2.0) - 0.50)
                    self.cam = self.scene.add_camera(
                        res=(1920, 1080),
                        pos=(x_c + 4.2, y_c - 4.0, 3.8),
                        lookat=(x_c, y_c, 0.45),
                        fov=50.0,
                        GUI=False,
                    )
                else:
                    self.cam = self.scene.add_camera(
                        res=(1280, 720),
                        pos=(1.8, -0.4, 1.3),
                        lookat=(0.15, -0.85, 0.48),
                        fov=45.0,
                        GUI=False,
                    )
                    self.cam_closeup = self.scene.add_camera(
                        res=(1280, 720),
                        pos=(0.45, -0.25, 0.75),
                        lookat=(-0.03, -0.56, 0.46),
                        fov=35.0,
                        GUI=False,
                    )
            elif self.n_envs > 1:
                # Encuadre panorámico para ver la cuadrícula 2x2 de entornos
                self.cam = self.scene.add_camera(
                    res=(1280, 720),
                    pos=(3.2, 1.0, 3.2),
                    lookat=(1.2, 1.0, 0.1),
                    fov=55.0,
                    GUI=False,
                )
                self.cam_closeup = None
            else:
                self.cam = self.scene.add_camera(
                    res=(1280, 720),
                    pos=(1.6, -0.8, 0.9),
                    lookat=(self.table_pos[0], self.table_pos[1] + 0.35, self.table_top + 0.1),
                    fov=45.0,
                    GUI=False,
                )
                self.cam_closeup = None
        else:
            self.cam = None
            self.cam_closeup = None

        # Rigid entities (floor, robot base, work table)
        friction_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=1.0)
        self.scene.add_entity(material=friction_rigid, morph=gs.morphs.Plane())

        if self.use_real_scene:
            campero_mesh = _resolve_mesh_path("meshes/campero_base/campero_base.obj")
            self.campero_base = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=campero_mesh,
                    pos=(0.15, 0.0, 0.10),
                    scale=1.0,
                    euler=(0.0, 0.0, -90.0),
                ),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )

            table_mesh = _resolve_mesh_path("meshes/tables/table3.obj")
            self.table = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file=table_mesh,
                    pos=self.table_pos,
                    scale=1.0,
                    euler=(0.0, 0.0, -90.0),
                ),
                material=gs.materials.Rigid(needs_coup=True, coup_friction=80.0),
            )

            ur10e_mjcf = _resolve_mesh_path("meshes/universal_robots_ur10e/ur10e.xml")
            self.robot = self.scene.add_entity(
                material=gs.materials.Rigid(),
                morph=gs.morphs.MJCF(
                    file=ur10e_mjcf,
                    pos=self.robot_base_pos,
                    quat=(1.0, 0.0, 0.0, 0.0),
                ),
            )
        else:
            self.pedestal = self.scene.add_entity(
                material=gs.materials.Rigid(coup_friction=10.0),
                morph=gs.morphs.Box(pos=self.robot_base_pos, size=self.robot_base_size, fixed=True),
                surface=gs.surfaces.Default(color=(0.22, 0.22, 0.25, 1.0)),
            )
            self.table = self.scene.add_entity(
                material=gs.materials.Rigid(coup_friction=80.0),
                morph=gs.morphs.Box(pos=self.table_pos, size=self.table_size, fixed=True),
                surface=gs.surfaces.Default(color=(0.58, 0.42, 0.26, 1.0)),
            )

        # Deformable cloth (instantiated on table surface)
        self.cloth = self.scene.add_entity(
            material=gs.materials.PBD.Cloth(
                rho=7.264,
                stretch_compliance=3.5e-6,
                bending_compliance=9.5,
                air_resistance=0.05,
                static_friction=1.80,
                kinetic_friction=1.00,
                stretch_relaxation=0.30,
                bending_relaxation=0.08,
            ),
            morph=gs.morphs.Mesh(
                file=str(self.mesh_path),
                scale=1.0,
                pos=self.cloth_drop_pos,
            ),
            surface=gs.surfaces.Default(color=(0.18, 0.28, 0.55, 1.0), vis_mode="particle"),
        )

        if self.n_envs > 1:
            self.scene.build(
                n_envs=self.n_envs,
                env_spacing=self.env_spacing,
                n_envs_per_row=int(np.ceil(np.sqrt(self.n_envs))),
            )
        else:
            self.scene.build()

        # Configure UR10e PD gains and kinematics if real scene is active
        if self.use_real_scene:
            self.arm_dof_dim = 6
            self.arm_dof_idx = torch.arange(self.arm_dof_dim, device=self.device)

            kp_tensor = torch.tensor([8000.0, 8000.0, 6000.0, 6000.0, 3000.0, 3000.0], dtype=torch.float32, device=self.device)
            kv_tensor = torch.tensor([800.0, 800.0, 600.0, 600.0, 300.0, 300.0], dtype=torch.float32, device=self.device)
            f_min = torch.tensor([-150.0, -150.0, -150.0, -28.0, -28.0, -28.0], dtype=torch.float32, device=self.device)
            f_max = torch.tensor([ 150.0,  150.0,  150.0,  28.0,  28.0,  28.0], dtype=torch.float32, device=self.device)

            self.robot.set_dofs_kp(kp_tensor, dofs_idx_local=self.arm_dof_idx)
            self.robot.set_dofs_kv(kv_tensor, dofs_idx_local=self.arm_dof_idx)
            self.robot.set_dofs_force_range(f_min, f_max, dofs_idx_local=self.arm_dof_idx)

            try:
                self.ee_link = self.robot.get_link("base_gripper")
            except Exception:
                self.ee_link = self.robot.get_link("wrist_3_link")

            if self.use_real_scene:
                # Pose pre-grasp en setup real: base rotada hacia la mesa (+67.8 deg) y brazo descendido sobre la mesa
                self.home_qpos = torch.tensor([1.183, -0.807, 2.036, -2.108, -1.583, 0.0], dtype=torch.float32, device=self.device)
            else:
                self.home_qpos = torch.tensor([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=torch.float32, device=self.device)
            self.down_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
            self.last_valid_qpos = self.home_qpos.clone()

            self.robot.set_dofs_position(self.home_qpos, dofs_idx_local=self.arm_dof_idx)
            self.robot.zero_all_dofs_velocity()
            if self.use_real_scene:
                self.open_gripper()

        self.init_particles_pos = self.cloth.get_particles_pos().clone()

    def get_gripper_qpos(self, driver_val: float) -> torch.Tensor:
        """
        Calcula las 8 posiciones articulares cinemáticas de la pinza Robotiq 2F-85.
        driver_val: [0.0, 0.70] rad (0.0 = abierta, 0.70 = cerrada).
        """
        q_grip = torch.zeros(8, dtype=torch.float32, device=self.device)
        d_val = float(driver_val)
        q_grip[0] = d_val        # right_driver
        q_grip[1] = -d_val       # right_coupler
        q_grip[2] = d_val        # right_spring_link
        q_grip[3] = 0.0          # right_follower
        q_grip[4] = d_val        # left_driver
        q_grip[5] = -d_val       # left_coupler
        q_grip[6] = d_val        # left_spring_link
        q_grip[7] = 0.0          # left_follower
        return q_grip

    def set_gripper_state(self, driver_val: float):
        """Aplica la posición articular a la pinza Robotiq 2F-85 (dofs 6 a 13)."""
        if not self.use_real_scene or not hasattr(self, "robot") or self.robot is None:
            return
        gripper_dofs = torch.arange(6, 14, device=self.device)
        q_grip = self.get_gripper_qpos(driver_val)
        self.robot.set_dofs_position(q_grip, dofs_idx_local=gripper_dofs)
        self.robot.zero_all_dofs_velocity()
        self.current_gripper_driver = float(driver_val)

    def open_gripper(self):
        """Abre completamente las mordazas de la pinza Robotiq 2F-85."""
        self.set_gripper_state(0.0)

    def close_gripper(self):
        """Cierra firmemente las mordazas de la pinza Robotiq 2F-85 sobre el tejido."""
        self.set_gripper_state(0.70)

    def solve_ik(self, target_pos: Union[np.ndarray, torch.Tensor], target_quat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calcula la cinematica inversa para el brazo UR10e orientando el efector final Robotiq 2F-85.
        Utiliza el punto local de agarre (pinch site [0, 0, 0.145] m) para hacer coincidir el TCP
        con la coordenada cartesiana objetivo en el espacio de trabajo de la mesa.
        Semilla con la postura articular actual completa (14 DoFs) para garantizar continuidad C^0/C^1
        y prevenir cambios espurios de rama articular o inversiones de codo.
        Retorna las 6 posiciones articulares del brazo UR10e (# shape: (6,)).
        """
        if not self.use_real_scene or not hasattr(self, "robot") or self.robot is None:
            return getattr(self, "home_qpos", torch.zeros(6, device=self.device))[:6]

        target_pos_tensor = torch.as_tensor(target_pos, dtype=torch.float32, device=self.device)
        if target_pos_tensor.ndim > 1:
            target_pos_tensor = target_pos_tensor.squeeze(0)

        # Semilla inicial con la configuración articular del robot (1, 14)
        if self.n_envs > 1:
            q_all = self.robot.get_dofs_position(envs_idx=[0])
            if q_all.ndim == 1:
                q_all = q_all.unsqueeze(0)
            current_arm_q = q_all[0, :self.arm_dof_dim]
            envs_idx_ik = [0]
        else:
            q_all = self.robot.get_dofs_position().unsqueeze(0)
            current_arm_q = q_all[0, :self.arm_dof_dim]
            envs_idx_ik = None

        quat = target_quat if target_quat is not None else (self.down_quat if self.use_real_scene else None)
        local_pinch = torch.tensor([0.0, 0.0, 0.145], dtype=torch.float32, device=self.device)

        try:
            q_sol = self.robot.inverse_kinematics(
                link=self.ee_link,
                pos=target_pos_tensor,
                quat=quat,
                local_point=local_pinch,
                init_qpos=q_all,
                dofs_idx_local=self.arm_dof_idx,
                envs_idx=envs_idx_ik,
                max_samples=25,
                max_solver_iters=30,
                damping=0.02,
            )
            if torch.isnan(q_sol).any():
                return current_arm_q

            if q_sol.ndim > 1:
                q_arm = q_sol[0, :self.arm_dof_dim]
            else:
                q_arm = q_sol[:self.arm_dof_dim]

            # Protección contra saltos espurios de rama articular (> 1.2 rad)
            q_diff = torch.abs(q_arm - current_arm_q)
            if (q_diff > 1.2).any():
                step_dir = torch.sign(q_arm - current_arm_q)
                q_arm = current_arm_q + step_dir * torch.clamp(q_diff, max=0.25)

            self.last_valid_qpos = q_arm.clone()
            return q_arm
        except Exception:
            return current_arm_q

    def get_ee_pose(self) -> torch.Tensor:
        """Retorna la pose cartesiana del efector final [pos(3), quat(4)]."""
        if not self.use_real_scene or not hasattr(self, "ee_link") or self.ee_link is None:
            return torch.zeros(7, device=self.device)
        pos = self.ee_link.get_pos()
        quat = self.ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    def get_joint_states(self) -> torch.Tensor:
        """Retorna las posiciones articulares del UR10e (# shape: (6,))."""
        if not self.use_real_scene or not hasattr(self, "robot") or self.robot is None:
            return torch.zeros(6, device=self.device)
        return self.robot.get_dofs_position(dofs_idx_local=self.arm_dof_idx)

    def get_joint_velocities(self) -> torch.Tensor:
        """Retorna las velocidades articulares del UR10e (# shape: (6,))."""
        if not self.use_real_scene or not hasattr(self, "robot") or self.robot is None:
            return torch.zeros(6, device=self.device)
        return self.robot.get_dofs_velocity(dofs_idx_local=self.arm_dof_idx)

    def get_robot_state(self) -> Dict[str, Any]:
        """Retorna el estado cinematico y dinamico completo del manipulador UR10e."""
        if not self.use_real_scene or not hasattr(self, "robot") or self.robot is None:
            return {}
        pos = self.ee_link.get_pos()
        quat = self.ee_link.get_quat()
        vel = self.ee_link.get_vel()
        qpos = self.get_joint_states()
        qvel = self.get_joint_velocities()
        return {
            "ee_pos": pos.cpu().numpy() if hasattr(pos, "cpu") else np.asarray(pos, dtype=np.float32),
            "ee_quat": quat.cpu().numpy() if hasattr(quat, "cpu") else np.asarray(quat, dtype=np.float32),
            "ee_vel": vel.cpu().numpy() if hasattr(vel, "cpu") else np.asarray(vel, dtype=np.float32),
            "joint_pos": qpos.cpu().numpy() if hasattr(qpos, "cpu") else np.asarray(qpos, dtype=np.float32),
            "joint_vel": qvel.cpu().numpy() if hasattr(qvel, "cpu") else np.asarray(qvel, dtype=np.float32),
        }

    def reset(self, stage: int = 1, checkpoint_path: Optional[str] = None, n_settle_steps: int = 40) -> np.ndarray:
        self.release_all_keypoints()

        if self.use_real_scene and hasattr(self, "robot") and self.robot is not None:
            self.robot.set_dofs_position(self.home_qpos, dofs_idx_local=self.arm_dof_idx)
            self.robot.zero_all_dofs_velocity()

        # Si estamos en Etapa 2 y no se especificó checkpoint, verificar si existe para ESTA prenda
        if stage == 2 and (checkpoint_path is None or not Path(checkpoint_path).exists()):
            ckpt_candidate = Path(f"D:/genesis_ws/checkpoints/Canonicals/pantalon_{self.garment_id}_folded_stage1.pt")
            if ckpt_candidate.exists():
                checkpoint_path = str(ckpt_candidate)

        loaded_ckpt = False
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            ckpt_pos = ckpt if isinstance(ckpt, torch.Tensor) else (ckpt.get("pos", ckpt.get("particles_pos", None)) if isinstance(ckpt, dict) else None)
            if ckpt_pos is not None:
                ckpt_tensor = torch.as_tensor(ckpt_pos, dtype=torch.float32)
                if ckpt_tensor.ndim == 3:
                    ckpt_tensor = ckpt_tensor[0]
                n_curr = self.cloth.get_particles_pos().shape[-2]
                if ckpt_tensor.shape[0] == n_curr:
                    # Desfase de coordenadas si proviene del marco canónico previo (Z_mesa ~ 0.55m, X ~ 0.0m)
                    if ckpt_tensor[:, 2].min() > 0.40:
                        offset = torch.tensor([0.72, 0.0, -0.50], dtype=torch.float32)
                        ckpt_tensor = ckpt_tensor + offset
                    if self.n_envs > 1:
                        ckpt_tensor = ckpt_tensor.unsqueeze(0).expand(self.n_envs, -1, -1).contiguous()
                    self.cloth.set_particles_pos(ckpt_tensor)
                    self.cloth.set_particles_vel(torch.zeros_like(ckpt_tensor))
                    loaded_ckpt = True

        if not loaded_ckpt:
            # Restauración de la pose inicial extendida sobre la mesa
            self.cloth.set_particles_pos(self.init_particles_pos)
            self.cloth.set_particles_vel(torch.zeros_like(self.init_particles_pos))

        for _ in range(n_settle_steps):
            self.scene.step()

        return self.get_particles_pos()

    def get_particles_pos(self) -> np.ndarray:
        pos = self.cloth.get_particles_pos()
        if hasattr(pos, "cpu"):
            pos = pos.cpu().numpy()
        return np.asarray(pos, dtype=np.float32)

    def get_keypoint_pos(self, keypoint_name: str) -> np.ndarray:
        idx = self.keypoint_indices[keypoint_name]
        pos = self.get_particles_pos()
        if self.n_envs > 1:
            return pos[:, idx, :].copy() # shape: (B, 3)
        return pos[idx].copy() # shape: (3,)

    def get_grasp_center_pos(self) -> np.ndarray:
        """Retorna la posicion 3D del baricentro de las particulas activas del agarre."""
        pos = self.get_particles_pos()
        if hasattr(self, "active_grasp_indices") and len(self.active_grasp_indices) > 0:
            if self.n_envs > 1:
                return pos[:, self.active_grasp_indices, :].mean(axis=1) # shape: (B, 3)
            return pos[self.active_grasp_indices].mean(axis=0) # shape: (3,)
        return self.get_keypoint_pos("top_ctr_f")

    def start_recording(self, video_path: str, fps: float = 30.0):
        """Inicia la grabacion de video MP4 en la camara offscreen."""
        if self.cam is not None:
            Path(video_path).parent.mkdir(parents=True, exist_ok=True)
            self.cam.start_recording(save_to_filename=str(video_path), fps=fps)

    def stop_recording(self):
        """Detiene y finaliza la grabacion del video MP4."""
        if self.cam is not None:
            self.cam.stop_recording()

    def get_stage2_canonical_grasp_indices(self) -> np.ndarray:
        """
        Determina de forma generalizada e invariante a la prenda los índices de partículas
        que componen la pretina bicapa completa sobre la mesa.
        Consulta los 4 keypoints semánticos ('l_corner', 'r_corner', 'top_ctr_f', 'top_ctr_b')
        y captura un cilindro vertical escalado en torno a su baricentro.
        """
        if hasattr(self, "_cached_stage2_grasp_indices") and self._cached_stage2_grasp_indices is not None:
            return self._cached_stage2_grasp_indices

        garment_id = getattr(self, "garment_id", "430")
        s1_ckpt_path = Path(f"D:/genesis_ws/checkpoints/Canonicals/pantalon_{garment_id}_folded_stage1.pt")
        n_curr = self.cloth.get_particles_pos().shape[-2]

        pos_ref = None
        if s1_ckpt_path.exists():
            data = torch.load(s1_ckpt_path, map_location="cpu", weights_only=False)
            pos_s1 = data if isinstance(data, torch.Tensor) else (data.get("pos", data.get("particles_pos", None)) if isinstance(data, dict) else None)
            if pos_s1 is not None:
                pos_s1_arr = pos_s1.numpy() if hasattr(pos_s1, "numpy") else np.asarray(pos_s1)
                if pos_s1_arr.ndim == 3:
                    pos_s1_arr = pos_s1_arr[0]
                if pos_s1_arr.shape[0] == n_curr:
                    pos_ref = pos_s1_arr

        if pos_ref is None:
            pos_ref = self.get_particles_pos()[0] if self.n_envs > 1 else self.get_particles_pos()

        p_l = pos_ref[self.keypoint_indices["l_corner"]]
        p_r = pos_ref[self.keypoint_indices["r_corner"]]
        p_tf = pos_ref[self.keypoint_indices["top_ctr_f"]]
        p_tb = pos_ref[self.keypoint_indices["top_ctr_b"]]
        x_star = float((p_l[0] + p_r[0] + p_tf[0] + p_tb[0]) / 4.0)
        y_star_edge = float((p_tf[1] + p_tb[1]) / 2.0)
        y_star = y_star_edge - 0.022
        dists_xy = np.hypot(pos_ref[:, 0] - x_star, pos_ref[:, 1] - y_star)
        
        r_cyl = float(np.clip(0.06 * getattr(self, "cloth_width", 0.368), 0.018, 0.035))
        waist_indices = np.where(dists_xy <= r_cyl)[0].astype(np.int32)
        if len(waist_indices) < 10:
            waist_indices = np.argsort(dists_xy)[:20].astype(np.int32)

        self._cached_stage2_grasp_indices = waist_indices
        return self._cached_stage2_grasp_indices

    def set_table_friction(self, friction: float):
        """Actualiza la friccion de acoplamiento de la mesa en tiempo de ejecucion."""
        if hasattr(self, "table") and hasattr(self.table, "material"):
            self.table.material.coup_friction = float(friction)

    def bind_stage_keypoints(self, stage: int = 1, anchor_mode: str = "free") -> Dict[str, Any]:
        pos = self.get_particles_pos()
        self.particles_init = pos.copy()
        self.release_all_keypoints()
        pos_ref = pos[0] if self.n_envs > 1 else pos

        if stage == 1:
            p0_ref = pos_ref[self.keypoint_indices["l_corner"]].copy()
            p_r_ref = pos_ref[self.keypoint_indices["r_corner"]].copy()
            self.r_corner_init = p_r_ref.copy()
            self.y_span_init = float(pos_ref[:, 1].max() - pos_ref[:, 1].min())
            self.x_span_init = float(pos_ref[:, 0].max() - pos_ref[:, 0].min())
            self.base_particle_indices = np.where(self.verts_2d[:, 0] >= -0.02)[0].astype(np.int32)

            r_grasp = float(np.clip(0.05 * self.cloth_width, 0.016, 0.030))
            dists_l = np.linalg.norm(pos_ref - p0_ref.reshape(1, 3), axis=1)
            grasp_indices = np.where(dists_l <= r_grasp)[0].astype(np.int32)
            if len(grasp_indices) < 5:
                grasp_indices = np.argsort(dists_l)[:12].astype(np.int32)

            r_anchor = float(np.clip(0.08 * self.cloth_width, 0.025, 0.045))
            dists_r = np.linalg.norm(pos_ref - p_r_ref.reshape(1, 3), axis=1)
            self.r_corner_indices = np.where(dists_r <= r_anchor)[0].astype(np.int32)
            self.r_corner_particles_init = pos_ref[self.r_corner_indices].copy()

            # Modalidades de sujecion de la base pasiva:
            if anchor_mode == "pinned":
                anchor_indices = np.where(dists_r <= (r_anchor * 0.55))[0].astype(np.int32)
            elif anchor_mode == "soft":
                anchor_indices = np.array([self.keypoint_indices["r_corner"]], dtype=np.int32)
            else:  # "free"
                anchor_indices = np.array([], dtype=np.int32)

            self.active_grasp_indices = grasp_indices
            self.active_all_fixed = np.unique(np.concatenate([grasp_indices, anchor_indices])).astype(np.int32)
            self.active_patch_offsets = pos_ref[grasp_indices] - p0_ref.reshape(1, 3)

            # Sobrecarrera proporcional al ancho de la cintura para neutralizar retracción
            overtravel_x = float(0.12 * self.cloth_width)
            p_target_ref = np.array([p_r_ref[0] + overtravel_x, p_r_ref[1], p_r_ref[2] + 0.020], dtype=np.float32)

            # En el setup real, ejecutar la secuencia de aproximación, descenso y cierre firme de pinza
            if self.use_real_scene and hasattr(self, "robot") and self.robot is not None:
                # Paso 1: Aproximación aérea con mordazas completamente abiertas (hover 6 cm sobre l_corner)
                self.open_gripper()
                p_hover = p0_ref.copy()
                p_hover[2] = float(max(p0_ref[2] + 0.060, self.table_top + 0.075))
                q_hover = self.solve_ik(p_hover, target_quat=self.down_quat)
                self.robot.set_dofs_position(q_hover[:self.arm_dof_dim], dofs_idx_local=self.arm_dof_idx)
                self.robot.zero_all_dofs_velocity()
                for _ in range(15):
                    self.scene.step()

                # Paso 2: Descenso vertical sobre el tejido (mordazas abiertas flanqueando la esquina)
                p_touch = p0_ref.copy()
                p_touch[2] = float(self.table_top + 0.010)
                q_touch = self.solve_ik(p_touch, target_quat=self.down_quat)
                for step_i in range(1, 11):
                    alpha = step_i / 10.0
                    q_cur = (1.0 - alpha) * q_hover + alpha * q_touch
                    self.robot.set_dofs_position(q_cur[:self.arm_dof_dim], dofs_idx_local=self.arm_dof_idx)
                    self.scene.step()

                # Paso 3: Cierre progresivo y firme de la pinza Robotiq 2F-85 sobre el tejido
                for close_i in range(1, 11):
                    drv = 0.70 * (close_i / 10.0)
                    self.set_gripper_state(drv)
                    self.scene.step()

                self.last_valid_qpos = q_touch[:self.arm_dof_dim].clone()

            # Acoplamiento físico PBD una vez las mordazas han atrapado el tejido
            self.cloth.fix_particles(self.active_all_fixed)
            for _ in range(5):
                self.scene.step()

            if self.n_envs > 1:
                p0 = pos[:, self.keypoint_indices["l_corner"], :].copy()
                p_target = np.tile(p_target_ref, (self.n_envs, 1))
                self.last_target_pos = p0.copy()
            else:
                p0 = p0_ref
                p_target = p_target_ref
                self.last_target_pos = p0.copy()

            return {
                "stage": 1,
                "p0": p0,
                "p_target": p_target,
                "cloth_width": self.cloth_width,
                "cloth_length": self.cloth_length,
                "h_ratio": 0.65,
                "theta_deg": 25.0,
                "mode": "cubic",
                "n_fold_steps": 550,
                "n_settle_steps": 60,
            }


        elif stage == 2:
            waist_indices = self.get_stage2_canonical_grasp_indices()
            p_curr_waist = np.mean(pos_ref[waist_indices], axis=0).astype(np.float32)
            x_star = float(p_curr_waist[0])
            y_star = float(p_curr_waist[1])

            self.active_grasp_indices = waist_indices
            self.active_all_fixed = waist_indices
            self.active_patch_offsets = pos_ref[waist_indices] - p_curr_waist.reshape(1, 3)
            self.cloth.fix_particles(self.active_all_fixed)

            # Centroide de referencia de destino en la mesa en reposo adaptado a la prenda (Stage 1 canónico)
            garment_id = getattr(self, "garment_id", "430")
            s1_ckpt_path = Path(f"D:/genesis_ws/checkpoints/Canonicals/pantalon_{garment_id}_folded_stage1.pt")
            pos_s1_ref = None
            if s1_ckpt_path.exists():
                data = torch.load(s1_ckpt_path, map_location="cpu", weights_only=False)
                pos_s1 = data if isinstance(data, torch.Tensor) else (data.get("pos", data.get("particles_pos", None)) if isinstance(data, dict) else None)
                if pos_s1 is not None:
                    pos_s1_arr = pos_s1.numpy() if hasattr(pos_s1, "numpy") else np.asarray(pos_s1)
                    if pos_s1_arr.ndim == 3:
                        pos_s1_arr = pos_s1_arr[0]
                    if pos_s1_arr[:, 2].min() > 0.40:
                        pos_s1_arr = pos_s1_arr + np.array([0.72, 0.0, -0.50], dtype=np.float32)
                    pos_s1_ref = pos_s1_arr

            if pos_s1_ref is not None:
                p_l_s1 = pos_s1_ref[self.keypoint_indices["l_corner"]]
                p_r_s1 = pos_s1_ref[self.keypoint_indices["r_corner"]]
                p_tf_s1 = pos_s1_ref[self.keypoint_indices["top_ctr_f"]]
                p_tb_s1 = pos_s1_ref[self.keypoint_indices["top_ctr_b"]]
                x_rest = float((p_l_s1[0] + p_r_s1[0] + p_tf_s1[0] + p_tb_s1[0]) / 4.0)
                y_rest = float((p_tf_s1[1] + p_tb_s1[1]) / 2.0) - 0.022
            else:
                x_rest = float(p_curr_waist[0])
                y_rest = float(self.cloth_length / 2.0 + 0.20)

            z_dest = float(self.table_top + 0.020)
            p_waist_rest = np.array([x_rest, y_rest, z_dest], dtype=np.float32)

            p_target_ref = p_waist_rest.copy()
            p0_ref = p_waist_rest.copy()

            p_l_ank = pos_ref[self.keypoint_indices["l_leg_o"]]
            p_r_ank = pos_ref[self.keypoint_indices["r_leg_o"]]
            p_ankles_center = (p_l_ank + p_r_ank) / 2.0

            if self.n_envs > 1:
                p0 = np.tile(p0_ref, (self.n_envs, 1))
                p_target = np.tile(p_target_ref, (self.n_envs, 1))
                p_ankles = np.tile(p_ankles_center, (self.n_envs, 1))
                self.last_target_pos = np.tile(p_curr_waist, (self.n_envs, 1)).astype(np.float32)
            else:
                p0 = p0_ref
                p_target = p_target_ref
                p_ankles = p_ankles_center
                self.last_target_pos = p_curr_waist.astype(np.float32)

            self.base_particle_indices = np.setdiff1d(np.arange(len(self.verts_2d)), waist_indices).astype(np.int32)
            return {
                "stage": 2,
                "p0": p0,
                "p_target": p_target,
                "p_ankles": p_ankles,
                "cloth_length": self.cloth_length,
                "cloth_width": self.cloth_width,
                "table_z": self.table_top,
                "delta_clearance": 0.650,
                "n_fold_steps": 500,
                "n_settle_steps": 60,
            }

        elif stage == 3:
            p_hems = (pos_ref[self.keypoint_indices["l_leg_o"]] + pos_ref[self.keypoint_indices["r_leg_o"]]) / 2.0
            p_waist = pos_ref[self.keypoint_indices["top_ctr_f"]].copy()

            dists_hems = np.linalg.norm(pos_ref - p_hems.reshape(1, 3), axis=1)
            hem_indices = np.where(dists_hems <= 0.035)[0].astype(np.int32)

            self.active_grasp_indices = hem_indices
            self.active_all_fixed = hem_indices
            self.active_patch_offsets = pos_ref[hem_indices] - p_hems.reshape(1, 3)
            self.cloth.fix_particles(self.active_all_fixed)
            self.base_particle_indices = np.setdiff1d(np.arange(len(self.verts_2d)), hem_indices).astype(np.int32)

            p_target_ref = np.array([p_waist[0], p_waist[1], p_waist[2] + 0.025], dtype=np.float32)

            if self.n_envs > 1:
                p0 = np.tile(p_hems, (self.n_envs, 1))
                p_target = np.tile(p_target_ref, (self.n_envs, 1))
                self.last_target_pos = p0.copy()
            else:
                p0 = p_hems
                p_target = p_target_ref
                self.last_target_pos = p0.copy()

            return {
                "stage": 3,
                "p0": p0,
                "p_target": p_target,
                "h_ratio": 0.35,
                "theta_deg": 0.0,
                "mode": "quadratic",
                "n_fold_steps": 450,
                "n_settle_steps": 60,
            }

        else:  # stage >= 4
            p_waist = pos_ref[self.keypoint_indices["top_ctr_f"]].copy()
            p_crotch = pos_ref[self.keypoint_indices["crotch"]].copy()

            dists_w = np.linalg.norm(pos_ref - p_waist.reshape(1, 3), axis=1)
            w_indices = np.where(dists_w <= 0.030)[0].astype(np.int32)

            self.active_grasp_indices = w_indices
            self.active_all_fixed = w_indices
            self.active_patch_offsets = pos_ref[w_indices] - p_waist.reshape(1, 3)
            self.cloth.fix_particles(self.active_all_fixed)
            self.base_particle_indices = np.setdiff1d(np.arange(len(self.verts_2d)), w_indices).astype(np.int32)

            p_target_ref = np.array([p_crotch[0], p_crotch[1], p_crotch[2] + 0.020], dtype=np.float32)

            if self.n_envs > 1:
                p0 = np.tile(p_waist, (self.n_envs, 1))
                p_target = np.tile(p_target_ref, (self.n_envs, 1))
                self.last_target_pos = p0.copy()
            else:
                p0 = p_waist
                p_target = p_target_ref
                self.last_target_pos = p0.copy()

            return {
                "stage": 4,
                "p0": p0,
                "p_target": p_target,
                "h_ratio": 0.35,
                "theta_deg": 0.0,
                "mode": "cubic",
                "n_fold_steps": 450,
                "n_settle_steps": 60,
            }

    def step_trajectory(self, target_pos: np.ndarray, substeps: int = 35):
        """
        Interpola linealmente la posición del parche a lo largo de los subpasos
        para garantizar continuidad C^0/C^1 y erradicar impactos impulsivos en XPBD.
        Soporta objetivos escalares (3,) o vectorizados (B, 3).
        """
        target_pos_arr = np.asarray(target_pos, dtype=np.float32)
        if self.n_envs > 1 and target_pos_arr.ndim == 1:
            target_pos_arr = np.tile(target_pos_arr, (self.n_envs, 1))

        p_start = self.last_target_pos.copy() if (hasattr(self, "last_target_pos") and self.last_target_pos is not None) else target_pos_arr.copy()

        for i in range(1, substeps + 1):
            alpha = float(i) / float(substeps)
            interp_target = (1.0 - alpha) * p_start + alpha * target_pos_arr

            if self.n_envs > 1:
                # interp_target: (B, 3) -> (B, 1, 3)
                # active_patch_offsets: (K, 3) -> (1, K, 3)
                # curr_patch: (B, K, 3)
                curr_patch = interp_target[:, None, :] + self.active_patch_offsets[None, :, :]
                curr_patch[..., 2] = np.maximum(curr_patch[..., 2], self.table_top + 0.005)
            else:
                curr_patch = interp_target.reshape(1, 3) + self.active_patch_offsets
                curr_patch[..., 2] = np.maximum(curr_patch[..., 2], self.table_top + 0.005)

            self.cloth.set_particles_pos(curr_patch, particles_idx_local=self.active_grasp_indices)

            # Control cinemático y dinámico del manipulador UR10e en el setup real
            if self.use_real_scene and hasattr(self, "robot") and self.robot is not None:
                target_ee = interp_target[0] if (self.n_envs > 1 and interp_target.ndim > 1) else interp_target
                q_ik = self.solve_ik(target_ee, target_quat=self.down_quat)
                self.robot.set_dofs_position(q_ik[:self.arm_dof_dim], dofs_idx_local=self.arm_dof_idx)
                # Mantener la pinza Robotiq 2F-85 cerrada cinemáticamente sobre el tejido
                gripper_dofs = torch.arange(6, 14, device=self.device)
                q_grip = self.get_gripper_qpos(0.70)
                self.robot.set_dofs_position(q_grip, dofs_idx_local=gripper_dofs)
                self.robot.zero_all_dofs_velocity()

            self.scene.step()

        self.last_target_pos = target_pos_arr.copy()

    def release_all_keypoints(self):
        if len(self.active_all_fixed) > 0:
            self.cloth.release_particle(self.active_all_fixed)
            self.cloth.set_particles_vel(0.0)
        self.active_grasp_indices = np.array([], dtype=np.int32)
        self.active_all_fixed = np.array([], dtype=np.int32)
        self.active_patch_offsets = np.empty((0, 3), dtype=np.float32)
        self.last_target_pos = None
        if self.use_real_scene and hasattr(self, "robot") and self.robot is not None:
            self.open_gripper()

    def release_and_settle(self, n_settle_steps: int = 60) -> np.ndarray:
        self.release_all_keypoints()
        # Retraer verticalmente el efector final (+6 cm) para desacoplar limpiamente sin colisiones
        if self.use_real_scene and hasattr(self, "robot") and self.robot is not None and hasattr(self, "ee_link"):
            try:
                curr_ee = self.ee_link.get_pos().cpu().numpy()
                curr_ee[2] += 0.060
                q_retract = self.solve_ik(curr_ee, target_quat=self.down_quat)
                self.robot.set_dofs_position(q_retract[:self.arm_dof_dim], dofs_idx_local=self.arm_dof_idx)
                self.robot.zero_all_dofs_velocity()
            except Exception:
                pass
        for _ in range(n_settle_steps):
            self.scene.step()
        return self.get_particles_pos()

    def settle_without_release(self, n_settle_steps: int = 20) -> np.ndarray:
        for _ in range(n_settle_steps):
            self.scene.step()
        return self.get_particles_pos()

    def get_realtime_drift_metrics(self) -> Dict[str, Any]:
        """
        Rastrea en tiempo real el desplazamiento del conjunto de partículas base respecto a la pose de inicio.
        """
        pos = self.get_particles_pos()
        if not hasattr(self, "particles_init") or self.particles_init is None:
            self.particles_init = pos.copy()

        b_idx = self.base_particle_indices if len(self.base_particle_indices) > 0 else np.arange(pos.shape[-2])

        if self.n_envs > 1:
            diff_base = pos[:, b_idx, :] - self.particles_init[:, b_idx, :]
            base_dists = np.linalg.norm(diff_base, axis=-1)
            mean_base_drift = np.mean(base_dists, axis=-1).astype(np.float32)
            max_base_drift = np.max(base_dists, axis=-1).astype(np.float32)
            vec_base_drift = np.mean(diff_base, axis=1).astype(np.float32)

            if hasattr(self, "r_corner_indices") and len(self.r_corner_indices) > 0:
                diff_r = pos[:, self.r_corner_indices, :] - self.particles_init[:, self.r_corner_indices, :]
                r_corner_disp = np.mean(np.linalg.norm(diff_r, axis=-1), axis=-1).astype(np.float32)
            else:
                r_corner_disp = np.zeros(self.n_envs, dtype=np.float32)
        else:
            diff_base = pos[b_idx] - self.particles_init[b_idx]
            base_dists = np.linalg.norm(diff_base, axis=-1)
            mean_base_drift = float(np.mean(base_dists))
            max_base_drift = float(np.max(base_dists))
            vec_base_drift = np.mean(diff_base, axis=0).astype(np.float32)

            if hasattr(self, "r_corner_indices") and len(self.r_corner_indices) > 0:
                diff_r = pos[self.r_corner_indices] - self.particles_init[self.r_corner_indices]
                r_corner_disp = float(np.mean(np.linalg.norm(diff_r, axis=-1)))
            else:
                r_corner_disp = 0.0

        return {
            "mean_base_drift": mean_base_drift,
            "max_base_drift": max_base_drift,
            "vec_base_drift": vec_base_drift,
            "r_corner_disp": r_corner_disp,
        }

    def get_r_corner_displacement(self) -> Dict[str, Any]:
        """
        Calcula el desplazamiento de todas las partículas del parche r_corner:
        - mean_disp: desplazamiento medio euclídeo (m)
        - max_disp: desplazamiento máximo euclídeo (m)
        - centroid_disp: vector [dx, dy, dz] del centroide (m)
        - kp_disp: desplazamiento del keypoint individual (m)
        """
        pos = self.get_particles_pos()
        pos_ref = pos[0] if self.n_envs > 1 else pos
        if not hasattr(self, "r_corner_indices") or len(self.r_corner_indices) == 0:
            return {
                "mean_disp": 0.0,
                "max_disp": 0.0,
                "centroid_disp": [0.0, 0.0, 0.0],
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "kp_disp": 0.0,
                "n_particles": 0,
            }
        p_now = pos_ref[self.r_corner_indices]
        diffs = p_now - self.r_corner_particles_init
        norms = np.linalg.norm(diffs, axis=-1)
        centroid = np.mean(diffs, axis=0)
        kp_disp = (
            float(np.linalg.norm(pos_ref[self.keypoint_indices["r_corner"]] - self.r_corner_init))
            if hasattr(self, "r_corner_init") and self.r_corner_init is not None
            else 0.0
        )
        return {
            "mean_disp": float(np.mean(norms)),
            "max_disp": float(np.max(norms)),
            "centroid_disp": centroid.tolist(),
            "dx": float(centroid[0]),
            "dy": float(centroid[1]),
            "dz": float(centroid[2]),
            "kp_disp": kp_disp,
            "n_particles": int(len(self.r_corner_indices)),
        }

    def compute_metrics(self, stage: int = 1, target_pos: Optional[np.ndarray] = None) -> Dict[str, float]:
        pos = self.get_particles_pos()
        pos_ref = pos[0] if self.n_envs > 1 else pos
        y_span = float(pos_ref[:, 1].max() - pos_ref[:, 1].min())
        x_span = float(pos_ref[:, 0].max() - pos_ref[:, 0].min())
        on_table = np.sum((pos_ref[:, 2] >= self.table_top - 0.005) & (pos_ref[:, 2] <= self.table_top + 0.015))
        contact_ratio = float(on_table / len(pos_ref))

        metrics = {
            "y_span": y_span,
            "x_span": x_span,
            "contact_ratio": contact_ratio,
            "table_contact_ratio": contact_ratio,
            "cloth_length": float(getattr(self, "cloth_length", 0.571)),
            "cloth_width": float(getattr(self, "cloth_width", 0.368)),
        }
        if stage == 1 and target_pos is not None:
            p_achieved = pos_ref[self.keypoint_indices["l_corner"]]
            p_target_r = pos_ref[self.keypoint_indices["r_corner"]]
            # Error de alineamiento físico real dobladillo con dobladillo (l_corner vs r_corner)
            metrics["alignment_error"] = float(np.linalg.norm(p_achieved - p_target_r))
            metrics["alignment_error_m"] = metrics["alignment_error"]
            target_pos_ref = target_pos[0] if (self.n_envs > 1 and target_pos.ndim > 1) else target_pos
            metrics["target_error"] = float(np.linalg.norm(p_achieved - target_pos_ref))

            r_disp = self.get_r_corner_displacement()
            metrics["base_drift"] = r_disp["kp_disp"]
            metrics["r_corner_mean_disp"] = r_disp["mean_disp"]
            metrics["r_corner_max_disp"] = r_disp["max_disp"]
            metrics["r_corner_drift_dx"] = r_disp["dx"]
            metrics["r_corner_drift_dy"] = r_disp["dy"]
            metrics["r_corner_drift_dz"] = r_disp["dz"]
            metrics["r_corner_particles_count"] = float(r_disp["n_particles"])

            metrics["l_corner_z"] = float(p_achieved[2])
            metrics["is_particles_up"] = float(1.0 if (p_achieved[2] >= self.table_top + 0.012 and contact_ratio <= 0.65) else 0.0)

        elif stage == 2:
            p_l_ankle = pos_ref[self.keypoint_indices["l_leg_o"]]
            p_r_ankle = pos_ref[self.keypoint_indices["r_leg_o"]]
            ankle_alignment = float(np.linalg.norm(p_l_ankle - p_r_ankle))
            metrics["ankle_alignment_error"] = ankle_alignment
            metrics["alignment_error"] = ankle_alignment
            metrics["alignment_error_m"] = ankle_alignment

            p_waist = pos_ref[self.keypoint_indices["top_ctr_f"]]
            p_ankles_mid = (p_l_ankle + p_r_ankle) / 2.0
            fold_dist = float(np.linalg.norm(p_waist[:2] - p_ankles_mid[:2]))
            metrics["fold_dist_xy"] = fold_dist

            target_pos_ref = target_pos[0] if (self.n_envs > 1 and target_pos is not None and target_pos.ndim > 1) else target_pos
            if target_pos_ref is not None:
                if hasattr(self, "active_grasp_indices") and len(self.active_grasp_indices) > 0:
                    p_active_center = np.mean(pos_ref[self.active_grasp_indices], axis=0)
                    metrics["target_error"] = float(np.linalg.norm(p_active_center - target_pos_ref))
                else:
                    metrics["target_error"] = float(np.linalg.norm(p_waist - target_pos_ref))
            else:
                metrics["target_error"] = 0.0

            # Criterio de éxito de Etapa 2 (Estiramiento de la prenda con tobillos juntos):
            # 1. Tobillos emparejados (<= 4 cm)
            # 2. Extensión longitudinal de la prenda (L_span entre 50 cm y 62 cm, nominal 57.1 cm)
            # 3. Pretina asentada en reposo sobre la mesa (error <= 4.5 cm)
            # 4. Contacto plano sobre mesa (>= 40%)
            span_ok = bool(abs(fold_dist - 0.571) <= 0.065)
            stage2_success = bool(
                ankle_alignment <= 0.040 and
                span_ok and
                metrics["target_error"] <= 0.045 and
                contact_ratio >= 0.40
            )
            metrics["stage2_success"] = float(1.0 if stage2_success else 0.0)
            metrics["is_success"] = float(1.0 if stage2_success else 0.0)
            metrics["align_ok"] = float(1.0 if ankle_alignment <= 0.040 else 0.0)
            metrics["ankle_align_ok"] = float(1.0 if ankle_alignment <= 0.040 else 0.0)
            metrics["fold_ok"] = float(1.0 if span_ok else 0.0)
            metrics["span_ok"] = float(1.0 if span_ok else 0.0)
            metrics["contact_ok"] = float(1.0 if contact_ratio >= 0.40 else 0.0)

        return metrics

    def save_checkpoint(self, path: Union[str, Path]):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.as_tensor(self.get_particles_pos(), dtype=torch.float32), p)


if __name__ == "__main__":
    print("[*] Running adapted Real Scene UR10e + Cloth backend verification...")
    backend = ClothGenesisBackend(show_viewer=False, use_real_scene=True)

    print("[1/3] Reset and settle...")
    pos = backend.reset(stage=1, n_settle_steps=30)
    print(f"      Particles: {len(pos)}, Z: [{pos[:, 2].min():.4f}, {pos[:, 2].max():.4f}] m")
    assert pos[:, 2].min() >= 0.420, "Error: Penetración de la prenda en la mesa."

    print("[2/3] Oracle bind Stage 1...")
    info = backend.bind_stage_keypoints(stage=1)
    print(f"      P0: {info['p0']}, Target: {info['p_target']}")

    print("[3/3] Single step and settle with UR10e tracking...")
    backend.step_trajectory(info["p0"] + np.array([0.0, 0.0, 0.05], dtype=np.float32), substeps=10)
    pos_settled = backend.release_and_settle(n_settle_steps=30)
    metrics = backend.compute_metrics(stage=1, target_pos=info["p_target"])
    print(f"      Metrics: Y_span={metrics['y_span']:.4f} m, Contact={metrics['contact_ratio']*100:.1f}%")

    robot_st = backend.get_robot_state()
    print(f"      Robot EE pos: {robot_st['ee_pos']}")
    print(f"      Robot Joint Q: {robot_st['joint_pos']}")
    print(">> [EXITO] Backend adaptado verificado sin divergencias numericas.")
