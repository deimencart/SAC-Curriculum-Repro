"""
dynamic_folding_env.py
======================
Gymnasium Environment for Multi-Stage Dynamic Cloth Folding in Genesis World (PBD).
"""

from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from envs.cloth_genesis_backend import ClothGenesisBackend
    from envs.bezier_geometry import BezierTrajectory, Stage2WaistDrapeTrajectory, Stage2UnifiedFoldTrajectory
except ImportError:
    try:
        from phase_5_gym_env.cloth_curriculum_rl.envs.cloth_genesis_backend import ClothGenesisBackend
        from phase_5_gym_env.cloth_curriculum_rl.envs.bezier_geometry import BezierTrajectory, Stage2WaistDrapeTrajectory, Stage2UnifiedFoldTrajectory
    except ImportError:
        from cloth_genesis_backend import ClothGenesisBackend
        from bezier_geometry import BezierTrajectory, Stage2WaistDrapeTrajectory, Stage2UnifiedFoldTrajectory



class DynamicFoldingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        garment_id: Union[int, str] = 430,
        dataset_dir: str = "D:/genesis_ws/FoldNet_2D/mesh/trousers",
        randomize_garment: bool = False,
        garment_pool: Optional[list] = None,
        show_viewer: bool = False,
        enable_camera: bool = False,
        rl_horizon_steps: int = 20,
        substeps_per_action: int = 35,
        action_clip_m: float = 0.03,
        delta_threshold: float = 0.08,
        n_envs: int = 1,
        randomize_theta: bool = True,
        theta_range: Tuple[float, float] = (15.0, 45.0),
        randomize_h_ratio: bool = True,
        h_ratio_range: Tuple[float, float] = (0.40, 0.75),
        release_at_end: bool = True,
        use_real_scene: bool = True,
    ):
        super().__init__()

        # Parameters of control and multi-garment generalisation
        self.garment_id = str(garment_id)
        self.dataset_dir = str(dataset_dir)
        self.randomize_garment = bool(randomize_garment)
        self.garment_pool = list(garment_pool) if garment_pool is not None else [430]
        self.show_viewer = show_viewer
        self.enable_camera = enable_camera
        self.rl_horizon_steps = rl_horizon_steps
        self.substeps_per_action = substeps_per_action
        self.action_clip_m = float(action_clip_m)
        self.delta_threshold = float(delta_threshold)
        self.n_envs = max(1, int(n_envs))
        self.randomize_theta = bool(randomize_theta)
        self.theta_range = (float(theta_range[0]), float(theta_range[1]))
        self.randomize_h_ratio = bool(randomize_h_ratio)
        self.h_ratio_range = (float(h_ratio_range[0]), float(h_ratio_range[1]))
        self.release_at_end = bool(release_at_end)
        self.use_real_scene = bool(use_real_scene)

        # Physical backend Genesis (soporta n_envs > 1 y cualquier prenda de FoldNet)
        self.backend = ClothGenesisBackend(
            garment_id=self.garment_id,
            dataset_dir=self.dataset_dir,
            show_viewer=self.show_viewer,
            enable_camera=self.enable_camera,
            n_envs=self.n_envs,
            use_real_scene=self.use_real_scene,
        )
        self.cloth_length = float(getattr(self.backend, "cloth_length", 0.571))
        self.cloth_width = float(getattr(self.backend, "cloth_width", 0.368))

        # 2. Espacio de acción de 5 dimensiones:
        # [0]: Delta x (corrección cartesiana fina en metros)
        # [1]: Delta y (corrección cartesiana fina en metros)
        # [2]: Delta z (corrección vertical residual en metros)
        # [3]: Delta h_lift (modulación de elevación aérea parabólica para peeling despegue)
        # [4]: Delta v (modulación de velocidad temporal y subpasos)
        # shape: (5,)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )
        # 3. Espacio de observación de 30 dimensiones:
        # - 7 keypoints x 3D (21)
        # - Progreso temporal tau (1)
        # - Error vectorial al waypoint 3D (3)
        # - Vector de deriva de la base 3D (3)
        # - Deriva media escalar de la base (1)
        # - Desplazamiento del parche r_corner (1)
        # shape: (30,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(30,),
            dtype=np.float32,
        )
        # 4. Variables de estado del episodio
        self.current_step = 0
        self.stage = 1
        self.traj: Optional[BezierTrajectory] = None
        self.stage_info: Dict[str, Any] = {}
        self.target_pos: Optional[np.ndarray] = None
        self.r_corner_drift_tol = 0.025
        self.r_drift_penalty_w = 2.0
        self.current_theta_deg = 25.0
        self.current_h_ratio = 0.55
        self.stage2_ankle_tol = 0.075

    def reset(
        self,
        stage: int = 1,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.current_step = 0
        self.stage = stage

        # 0. Gestion de cambio o aleatorizacion de prenda FoldNet
        target_gid = None
        if options and "garment_id" in options:
            target_gid = str(options["garment_id"])
        elif self.randomize_garment and len(self.garment_pool) > 0:
            target_gid = str(self.np_random.choice(self.garment_pool))

        if target_gid is not None and target_gid != self.garment_id:
            swapped = self.backend.load_garment(target_gid)
            if swapped:
                self.garment_id = target_gid
                self.cloth_length = float(getattr(self.backend, "cloth_length", 0.571))
                self.cloth_width = float(getattr(self.backend, "cloth_width", 0.368))

        # 1. Opciones curriculares (sujecion, cercania, friccion, horizonte)
        ckpt_path = options.get("checkpoint_path", None) if options else None
        anchor_mode = str(options.get("anchor_mode", "free")) if options else "free"
        friction = float(options.get("friction", 80.0)) if options else 80.0
        self.tau_start = float(options.get("tau_start", 0.0)) if options else 0.0
        self.tau = self.tau_start
        if options and "rl_horizon" in options:
            self.rl_horizon_steps = int(options["rl_horizon"])
        if options and "substeps" in options:
            self.substeps_per_action = int(options["substeps"])
        elif self.stage == 2:
            self.substeps_per_action = 60
        if options and "release_at_end" in options:
            self.release_at_end = bool(options["release_at_end"])
        self.stage2_ankle_tol = float(options.get("ankle_tol", 0.075)) if options else 0.075
        # Parametros de la politica de restriccion de r_corner (escalan con el nivel curricular)
        self.r_corner_drift_tol = float(options.get("r_corner_drift_tol", 0.025)) if options else 0.025
        self.r_drift_penalty_w = float(options.get("r_drift_penalty_w", 2.0)) if options else 2.0

        # Reinicio fisico de la prenda sobre la mesa
        n_settle = 80 if (self.stage == 2 and self.tau_start >= 0.60) else 40
        self.backend.reset(stage=self.stage, checkpoint_path=ckpt_path, n_settle_steps=n_settle)
        self.backend.set_table_friction(friction)

        # 2. Anclaje de keypoints segun la fase y modo de sujecion
        self.stage_info = self.backend.bind_stage_keypoints(stage=self.stage, anchor_mode=anchor_mode)
        target_raw = self.stage_info["p_target"]
        self.target_pos = np.asarray(target_raw, dtype=np.float32)

        # 3. Construccion de la trayectoria nominal analitica de Bezier
        p0_raw = self.stage_info["p0"]
        p0_single = np.asarray(p0_raw[0] if self.n_envs > 1 else p0_raw, dtype=np.float32)
        target_single = np.asarray(target_raw[0] if self.n_envs > 1 else target_raw, dtype=np.float32)

        if self.stage == 2:
            cloth_length = float(self.stage_info.get("cloth_length", getattr(self.backend, "cloth_length", 0.571)))
            delta_clearance = float(self.stage_info.get("delta_clearance", 0.080))
            p_ankles_raw = self.stage_info.get("p_ankles", None)
            p_ankles_single = np.asarray(p_ankles_raw[0] if (self.n_envs > 1 and p_ankles_raw is not None and getattr(p_ankles_raw, "ndim", 1) > 1) else p_ankles_raw, dtype=np.float32) if p_ankles_raw is not None else None
            self.traj = Stage2UnifiedFoldTrajectory(
                p0=p0_single,
                p_target=target_single,
                p_ankles=p_ankles_single,
                cloth_length=cloth_length,
                table_z=self.backend.table_top,
                delta_clearance=delta_clearance,
            )
        else:
            # Domain Randomization basado en De Gusseme et al. (2022) para Etapa 1:
            # Sweet Spot robusto: theta in [15.0, 45.0] deg, h_ratio in [0.40, 0.75]
            if options and "theta_deg" in options:
                theta_deg = float(options["theta_deg"])
            elif self.randomize_theta and self.stage == 1:
                theta_deg = float(self.np_random.uniform(self.theta_range[0], self.theta_range[1]))
            else:
                theta_deg = float(self.stage_info.get("theta_deg", 25.0))
            self.current_theta_deg = theta_deg

            if options and "h_ratio" in options:
                h_ratio = float(options["h_ratio"])
            elif self.randomize_h_ratio and self.stage == 1:
                h_ratio = float(self.np_random.uniform(self.h_ratio_range[0], self.h_ratio_range[1]))
            else:
                h_ratio = float(self.stage_info.get("h_ratio", 0.55))
            self.current_h_ratio = h_ratio

            mode = str(self.stage_info.get("mode", "cubic"))

            self.traj = BezierTrajectory(
                p0=p0_single,
                p_target=target_single,
                h_ratio=h_ratio,
                theta_deg=theta_deg,
                mode=mode,
            )

        # Si hay currículo de cercanía en Stage 1 (tau_start > 0), posicionar suavemente el manipulador a lo largo de la trayectoria.
        # En Stage 2, las poses intermedias (Pose B o Pose C) ya cargan la tela directamente en esa coordenada física.
        if self.stage == 1 and self.tau_start > 0.0:
            n_ramp = int(np.clip(self.tau_start * 20, 5, 15))
            for i in range(1, n_ramp + 1):
                tau_i = float(self.tau_start * (i / float(n_ramp)))
                p_i = self.traj.get_position(tau_i)
                self.backend.step_trajectory(target_pos=p_i, substeps=15)

        # Flags de control para el doble asentamiento gravitatorio (Stage 2)
        self._air_settled = bool(self.tau_start >= 0.25)
        self._touchdown_settled = bool(self.tau_start >= 0.65)

        # 4. Observacion inicial
        obs = self._get_obs(tau=self.tau)
        info = {
            "stage": self.stage,
            "target_pos": self.target_pos.copy(),
            "p0": np.asarray(p0_raw).copy(),
            "anchor_mode": anchor_mode,
            "friction": friction,
            "tau_start": self.tau_start,
            "r_corner_drift_tol": self.r_corner_drift_tol,
            "r_drift_penalty_w": self.r_drift_penalty_w,
            "theta_deg": self.current_theta_deg,
            "h_ratio": self.current_h_ratio,
        }
        if hasattr(self.backend, "get_robot_state"):
            info["robot_state"] = self.backend.get_robot_state()
        return obs, info



    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, Any, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        action_arr = np.asarray(action, dtype=np.float32)

        # 1. Modulacion adaptativa de velocidad, altura y avance temporal:
        # Accion 5D: [dx, dy, dz, d_hlift, dv]
        if action_arr.shape[-1] >= 5:
            dx = action_arr[..., 0] * self.action_clip_m
            dy = action_arr[..., 1] * self.action_clip_m
            dz = action_arr[..., 2] * 0.05
            delta_h = np.clip((action_arr[..., 3] + 1.0) / 2.0, 0.0, 1.0) * 0.15
            delta_v = action_arr[..., 4] * 0.5
            action_xyz = np.stack([dx, dy, dz], axis=-1)
            v_factor = float(np.clip(1.0 + np.mean(delta_v), 0.5, 1.5))
            h_lift_boost = float(np.mean(delta_h))
        elif action_arr.shape[-1] == 4:
            action_xyz = action_arr[..., :3]
            delta_v = float(np.mean(action_arr[..., 3]))
            v_factor = float(np.clip(1.0 + 0.5 * delta_v, 0.5, 1.5))
            h_lift_boost = 0.0
        else:
            action_xyz = action_arr
            v_factor = 1.0
            h_lift_boost = 0.0

        d_tau = ((1.0 - self.tau_start) / max(1, self.rl_horizon_steps)) * v_factor
        self.tau = min(1.0, self.tau + d_tau)
        tau = self.tau

        # 2. Posicion nominal Bezier C^2 con boost de elevacion parabolico para peeling (solo en Etapa 1)
        x_nominal_single = self.traj.get_position(tau).copy() # shape: (3,)
        if self.stage == 1:
            parabola_factor = float(4.0 * tau * (1.0 - tau))
            x_nominal_single[2] += h_lift_boost * parabola_factor

        # 3. Accion residual acotada
        action_bounds = np.array([self.action_clip_m, self.action_clip_m, 0.08], dtype=np.float32)
        min_safe_z = float(self.backend.table_top + 0.020)
        if self.n_envs > 1:
            x_nominal = np.tile(x_nominal_single, (self.n_envs, 1)) # shape: (B, 3)
            action_clipped = np.clip(action_xyz, -action_bounds, action_bounds)
            x_cmd = x_nominal + action_clipped
            x_cmd[:, 2] = np.maximum(x_cmd[:, 2], min_safe_z)
        else:
            action_clipped = np.clip(action_xyz, -action_bounds, action_bounds)
            x_cmd = x_nominal_single + action_clipped
            x_cmd[2] = max(float(x_cmd[2]), min_safe_z)

        # 4. Integracion fisica en Genesis XPBD con cadencia adaptativa
        active_substeps = int(np.clip(self.substeps_per_action / v_factor, 20, 90))
        self.backend.step_trajectory(target_pos=x_cmd, substeps=active_substeps)

        # En Etapa 2: doble pausa de asentamiento gravitatorio (Fase b aérea tau >= 0.25 y Fase d frontal tau >= 0.65)
        if self.stage == 2 and not self._air_settled and tau >= 0.25:
            self._air_settled = True
            self.backend.step_trajectory(target_pos=x_cmd, substeps=200)

        if self.stage == 2 and not self._touchdown_settled and tau >= 0.65:
            self._touchdown_settled = True
            self.backend.step_trajectory(target_pos=x_cmd, substeps=200)

        # 5. Observación tras la física
        obs = self._get_obs(tau)

        # 6. Cálculo de distancia y recompensa cóncava (Li et al., 2025)
        if self.stage == 1:
            p_active = self.backend.get_keypoint_pos("l_corner")
        elif self.stage == 2:
            p_active = self.backend.get_grasp_center_pos()
        else:
            p_active = (
                self.backend.get_keypoint_pos("l_leg_o")
                + self.backend.get_keypoint_pos("r_leg_o")
            ) / 2.0

        eps = 1e-7
        table_top_z = self.backend.table_top

        # Metricas de deriva de la base y r_corner en tiempo real
        drift_data = self.backend.get_realtime_drift_metrics()
        mean_base_drift = drift_data["mean_base_drift"]
        r_corner_disp = drift_data["r_corner_disp"]

        # Follow the Object (Kılınç & Montana, 2020) + Recompensas por Etapa:
        if self.stage == 2:
            p_l_ankle = self.backend.get_keypoint_pos("l_leg_o")
            p_r_ankle = self.backend.get_keypoint_pos("r_leg_o")
            if self.n_envs > 1:
                dist_to_waypoint = np.linalg.norm(p_active - x_nominal_single[None, :], axis=-1)
                d_norm_wp = dist_to_waypoint / (self.delta_threshold + eps)
                r_track = np.where(d_norm_wp > 1.0, -1.0, 1.0 - (d_norm_wp ** 0.4)).astype(np.float32)

                dist_to_final = np.linalg.norm(p_active - self.target_pos, axis=-1)
                r_final = (4.0 * np.exp(- dist_to_final / (self.delta_threshold + eps))).astype(np.float32)

                ankle_scale = float(0.10 * self.cloth_width + eps)
                cuff_scale = float(0.12 * self.cloth_width + eps)
                stretch_scale = float(0.15 * self.cloth_length + eps)
                length_margin = float(max(0.065, 0.10 * self.cloth_length))

                ankle_dist = np.linalg.norm(p_l_ankle - p_r_ankle, axis=-1)
                r_ankle = np.clip(2.0 * np.exp(- ankle_dist / ankle_scale) - 0.5, -1.0, 2.0).astype(np.float32)

                p_ankles_mid = (p_l_ankle + p_r_ankle) / 2.0
                span_dist = np.linalg.norm(p_active[..., :2] - p_ankles_mid[..., :2], axis=-1)
                r_stretch = np.clip(1.5 * (1.0 - np.abs(span_dist - self.cloth_length) / stretch_scale), -1.5, 1.5).astype(np.float32)

                r_smooth = (- 1.0 * np.maximum(0.0, v_factor - 1.2)).astype(np.float32)

                p_l_cuff = self.backend.get_keypoint_pos("l_corner")
                p_r_cuff = self.backend.get_keypoint_pos("r_corner")
                cuff_dist = np.linalg.norm(p_l_cuff - p_r_cuff, axis=-1)
                r_cuff = np.clip(1.5 * np.exp(- cuff_dist / cuff_scale) - 0.5, -1.0, 1.5).astype(np.float32)

                if tau >= 0.85:
                    z_waist = p_active[..., 2] - table_top_z
                    r_landing = np.where((z_waist >= 0.010) & (z_waist <= 0.035), 2.0, np.where(z_waist > 0.050, -3.0 * (z_waist - 0.035), -0.5)).astype(np.float32)
                    reward = (0.3 * r_track + 1.0 * r_final + 1.2 * r_ankle + 1.0 * r_cuff + 1.5 * r_stretch + r_landing + r_smooth).astype(np.float32)
                elif tau >= 0.70:
                    reward = (0.5 * r_track + 0.8 * r_final + 1.2 * r_ankle + 1.0 * r_cuff + 1.2 * r_stretch + r_smooth).astype(np.float32)
                elif tau >= 0.45:
                    reward = (r_track + 2.0 * r_ankle + 0.8 * r_cuff + r_smooth).astype(np.float32)
                else:
                    reward = (r_track + 2.5 * r_ankle + 0.8 * r_cuff + r_smooth).astype(np.float32)

                dist_to_target = dist_to_final
                d_norm = dist_to_final / (self.delta_threshold + eps)
                span_ok = np.abs(span_dist - self.cloth_length) <= length_margin
                is_success = (dist_to_final <= max(0.045, 0.10 * self.cloth_width)) & (ankle_dist <= max(self.stage2_ankle_tol, 0.045)) & (cuff_dist <= max(0.050, 0.12 * self.cloth_width)) & span_ok
            else:
                ankle_scale = float(0.10 * self.cloth_width + eps)
                cuff_scale = float(0.12 * self.cloth_width + eps)
                stretch_scale = float(0.15 * self.cloth_length + eps)
                length_margin = float(max(0.065, 0.10 * self.cloth_length))

                dist_to_waypoint = float(np.linalg.norm(p_active - x_nominal_single))
                d_norm_wp = dist_to_waypoint / (self.delta_threshold + eps)
                r_track = float(-1.0 if d_norm_wp > 1.0 else 1.0 - (d_norm_wp ** 0.4))

                dist_to_final = float(np.linalg.norm(p_active - self.target_pos))
                r_final = float(4.0 * np.exp(- dist_to_final / (self.delta_threshold + eps)))

                ankle_dist = float(np.linalg.norm(p_l_ankle - p_r_ankle))
                r_ankle = float(np.clip(2.0 * np.exp(- ankle_dist / ankle_scale) - 0.5, -1.0, 2.0))

                p_l_cuff = self.backend.get_keypoint_pos("l_corner")
                p_r_cuff = self.backend.get_keypoint_pos("r_corner")
                cuff_dist = float(np.linalg.norm(p_l_cuff - p_r_cuff))
                r_cuff = float(np.clip(1.5 * np.exp(- cuff_dist / cuff_scale) - 0.5, -1.0, 1.5))

                p_ankles_mid = (p_l_ankle + p_r_ankle) / 2.0
                span_dist = float(np.linalg.norm(p_active[:2] - p_ankles_mid[:2]))
                r_stretch = float(np.clip(1.5 * (1.0 - abs(span_dist - self.cloth_length) / stretch_scale), -1.5, 1.5))

                r_smooth = float(- 1.0 * max(0.0, v_factor - 1.2))

                if tau >= 0.85:
                    z_waist = float(p_active[2] - table_top_z)
                    r_landing = 2.0 if (0.010 <= z_waist <= 0.035) else (float(-3.0 * (z_waist - 0.035)) if z_waist > 0.050 else -0.5)
                    reward = float(0.3 * r_track + 1.0 * r_final + 1.2 * r_ankle + 1.0 * r_cuff + 1.5 * r_stretch + r_landing + r_smooth)
                elif tau >= 0.70:
                    reward = float(0.5 * r_track + 0.8 * r_final + 1.2 * r_ankle + 1.0 * r_cuff + 1.2 * r_stretch + r_smooth)
                elif tau >= 0.45:
                    reward = float(r_track + 2.0 * r_ankle + 0.8 * r_cuff + r_smooth)
                else:
                    reward = float(r_track + 2.5 * r_ankle + 0.8 * r_cuff + r_smooth)

                dist_to_target = dist_to_final
                d_norm = dist_to_final / (self.delta_threshold + eps)
                span_ok = bool(abs(span_dist - self.cloth_length) <= length_margin)
                is_success = bool((dist_to_final <= max(0.045, 0.10 * self.cloth_width)) and (ankle_dist <= max(self.stage2_ankle_tol, 0.045)) and (cuff_dist <= max(0.050, 0.12 * self.cloth_width)) and span_ok)

        else: # Stage 1 y otras
            if self.n_envs > 1:
                dist_to_waypoint = np.linalg.norm(p_active - x_nominal_single[None, :], axis=-1)
                d_norm_wp = dist_to_waypoint / (self.delta_threshold + eps)
                r_track = np.where(d_norm_wp > 1.0, -1.0, 1.0 - (d_norm_wp ** 0.4)).astype(np.float32)

                dist_to_final = np.linalg.norm(p_active - self.target_pos, axis=-1)
                r_final = (4.0 * np.exp(- dist_to_final / (self.delta_threshold + eps))).astype(np.float32)

                drift_ratio = r_corner_disp / (self.r_corner_drift_tol + eps)
                base_ratio = mean_base_drift / 0.05
                r_drift = (
                    - self.r_drift_penalty_w * np.tanh(drift_ratio)
                    - 2.0 * np.tanh(base_ratio)
                ).astype(np.float32)

                r_jerk = np.where(tau < 0.25, - 2.5 * np.maximum(0.0, v_factor - 0.75), 0.0).astype(np.float32)

                z_clearance = p_active[..., 2] - table_top_z
                mid_phase = (tau >= 0.15) and (tau <= 0.75)
                if mid_phase:
                    r_peel = np.where(z_clearance < 0.06, -0.4, np.clip(z_clearance / 0.18, 0.0, 1.0) * 0.4).astype(np.float32)
                else:
                    r_peel = np.zeros(self.n_envs, dtype=np.float32)

                if tau >= 0.80:
                    z_cuff = p_active[..., 2] - table_top_z
                    r_landing = np.where(
                        (z_cuff >= 0.006) & (z_cuff <= 0.030),
                        1.5,
                        np.where(z_cuff > 0.045, - 4.0 * (z_cuff - 0.030), -0.5)
                    ).astype(np.float32)
                else:
                    r_landing = np.zeros(self.n_envs, dtype=np.float32)

                if tau >= 0.80:
                    reward = (0.3 * r_track + 0.7 * r_final + r_drift + r_landing + r_jerk).astype(np.float32)
                else:
                    reward = (r_track + r_drift + r_peel + r_jerk).astype(np.float32)

                dist_to_target = dist_to_final
                d_norm = dist_to_final / (self.delta_threshold + eps)
                is_success = (dist_to_final <= self.delta_threshold)
            else:
                dist_to_waypoint = float(np.linalg.norm(p_active - x_nominal_single))
                d_norm_wp = dist_to_waypoint / (self.delta_threshold + eps)
                r_track = float(-1.0 if d_norm_wp > 1.0 else 1.0 - (d_norm_wp ** 0.4))

                dist_to_final = float(np.linalg.norm(p_active - self.target_pos))
                r_final = float(4.0 * np.exp(- dist_to_final / (self.delta_threshold + eps)))

                drift_ratio = float(r_corner_disp) / (self.r_corner_drift_tol + eps)
                base_ratio = float(mean_base_drift) / 0.05
                r_drift = float(
                    - self.r_drift_penalty_w * np.tanh(drift_ratio)
                    - 2.0 * np.tanh(base_ratio)
                )

                r_jerk = float(- 2.5 * max(0.0, v_factor - 0.75)) if tau < 0.25 else 0.0

                z_clearance = float(p_active[2] - table_top_z)
                if 0.15 <= tau <= 0.75:
                    r_peel = -0.4 if z_clearance < 0.06 else min(0.4, max(0.0, z_clearance / 0.18) * 0.4)
                else:
                    r_peel = 0.0

                if tau >= 0.80:
                    z_cuff = float(p_active[2] - table_top_z)
                    if 0.006 <= z_cuff <= 0.030:
                        r_landing = 1.5
                    elif z_cuff > 0.045:
                        r_landing = float(- 4.0 * (z_cuff - 0.030))
                    else:
                        r_landing = -0.5
                else:
                    r_landing = 0.0

                if tau >= 0.80:
                    reward = float(0.3 * r_track + 0.7 * r_final + r_drift + r_landing + r_jerk)
                else:
                    reward = float(r_track + r_drift + r_peel + r_jerk)

                dist_to_target = dist_to_final
                d_norm = dist_to_final / (self.delta_threshold + eps)
                is_success = bool(dist_to_final <= self.delta_threshold)

        # 7. Criterios de parada y fusible anti-explosión
        terminated = bool(self.tau >= 0.999)
        max_steps = int(self.rl_horizon_steps * 2.2) if self.stage == 2 else self.rl_horizon_steps
        truncated = bool(self.current_step >= max_steps)

        # Fusible de seguridad física: truncar de inmediato si hay partículas eyectadas o NaNs
        particles_curr = self.backend.get_particles_pos()
        max_z_detected = float(np.max(particles_curr[..., 2]))
        safe_max_z = float(self.backend.table_top + 0.85)
        fuse_tripped = bool(max_z_detected > safe_max_z or np.isnan(particles_curr).any())
        if fuse_tripped:
            truncated = True
            reward = float(-500.0) if self.n_envs == 1 else np.full((self.n_envs,), -500.0, dtype=np.float32)
            is_success = False if self.n_envs == 1 else np.zeros((self.n_envs,), dtype=bool)

        drift_instant = self.backend.get_realtime_drift_metrics()

        info = {
            "dist_to_target": dist_to_target,
            "d_norm": d_norm,
            "tau": tau,
            "stage": self.stage,
            "is_success": is_success,
            "fuse_tripped": fuse_tripped,
            "max_z": max_z_detected,
            "mean_base_drift": float(np.mean(drift_instant["mean_base_drift"])),
            "max_base_drift": float(np.max(drift_instant["max_base_drift"])),
            "r_corner_mean_disp": float(np.mean(drift_instant["r_corner_disp"])),
            "v_factor": v_factor,
            "h_lift_boost": h_lift_boost,
            "theta_deg": self.current_theta_deg,
            "h_ratio": self.current_h_ratio,
        }
        if hasattr(self.backend, "get_robot_state"):
            info["robot_state"] = self.backend.get_robot_state()



        # Al completar el horizonte (por final de trayectoria o límite de pasos), asentamos la tela
        if terminated or truncated:
            if self.release_at_end:
                self.backend.release_and_settle(n_settle_steps=40)
            else:
                self.backend.settle_without_release(n_settle_steps=20)
            metrics = self.backend.compute_metrics(stage=self.stage, target_pos=self.target_pos)
            info.update(metrics)

            if self.stage == 1:
                align_err = float(metrics.get("alignment_error", 1.0))
                r_mean_drift = float(metrics.get("r_corner_mean_disp", metrics.get("base_drift", 1.0)))
                p_up = bool(metrics.get("is_particles_up", 0.0) > 0.5)
                cuff_dist = align_err
                
                # Tolerancias adaptativas a la envergadura transversal de la prenda
                cuff_tol = float(max(0.04, 0.12 * self.cloth_width))
                align_tol = float(max(self.delta_threshold, 0.12 * self.cloth_width))
                span_ok = bool(cuff_dist <= cuff_tol)
                align_ok = bool(align_err <= align_tol)
                drift_ok = bool(r_mean_drift <= self.r_corner_drift_tol)

                # Éxito físico compuesto por los 4 criterios rigurosos
                is_success = bool(align_ok and drift_ok and p_up and span_ok)
                info["is_success"] = is_success
                info["align_ok"] = align_ok
                info["drift_ok"] = drift_ok
                info["particles_up_ok"] = p_up
                info["span_ok"] = span_ok
                info["r_corner_drift_tol"] = self.r_corner_drift_tol
                info["r_drift_penalty_w"] = self.r_drift_penalty_w
                info["alignment_error_m"] = align_err

                # Bonus terminal por consecución del objetivo físico
                if not fuse_tripped:
                    if is_success:
                        term_bonus = 25.0
                    elif align_ok and span_ok:
                        term_bonus = 10.0
                    elif align_ok:
                        term_bonus = 5.0
                    else:
                        term_bonus = - 8.0 * float(np.clip(align_err / (0.30 * self.cloth_width + 1e-7), 0.0, 2.0))

                    if self.n_envs > 1:
                        reward = (reward + term_bonus).astype(np.float32)
                    else:
                        reward = float(reward + term_bonus)

            elif self.stage == 2:
                ankle_err = float(metrics.get("ankle_alignment_error", 1.0))
                fold_dist = float(metrics.get("fold_dist_xy", 1.0))
                target_err = float(metrics.get("target_error", 1.0))
                contact = float(metrics.get("contact_ratio", 0.0))

                # Tolerancia adaptativa al largo nominal de la prenda (10% de margen)
                length_margin = float(max(0.065, 0.10 * self.cloth_length))
                target_tol = float(max(0.045, 0.10 * self.cloth_width))

                ankle_ok = bool(ankle_err <= self.stage2_ankle_tol)
                span_ok = bool(abs(fold_dist - self.cloth_length) <= length_margin)
                target_ok = bool(target_err <= target_tol)
                contact_ok = bool(contact >= 0.40)

                # Éxito físico en Etapa 2: tobillos emparejados, envergadura longitudinal completa, pretina en reposo y contacto plano
                is_success = bool(ankle_ok and span_ok and target_ok and contact_ok)
                info["is_success"] = is_success
                info["align_ok"] = ankle_ok
                info["ankle_align_ok"] = ankle_ok
                info["fold_ok"] = span_ok
                info["span_ok"] = span_ok
                info["target_ok"] = target_ok
                info["contact_ok"] = contact_ok
                info["alignment_error_m"] = ankle_err
                info["fold_dist_xy"] = fold_dist
                info["target_error"] = target_err

                if not fuse_tripped:
                    if is_success:
                        term_bonus = 25.0
                    elif ankle_ok and span_ok:
                        term_bonus = 15.0
                    elif span_ok:
                        term_bonus = 8.0
                    elif ankle_ok:
                        term_bonus = 5.0
                    else:
                        term_bonus = - 8.0 * float(np.clip(abs(fold_dist - self.cloth_length) / (0.20 * self.cloth_length + 1e-7), 0.0, 2.0))

                    if self.n_envs > 1:
                        reward = (reward + term_bonus).astype(np.float32)
                    else:
                        reward = float(reward + term_bonus)

        return obs, reward, terminated, truncated, info

    def _get_obs(self, tau: float) -> np.ndarray:
        """
        Construye el vector de observación de 30 dimensiones:
        - 7 keypoints anatómicos x 3D = 21 floats
        - Progreso temporal tau normalizado = 1 float
        - Vector de error instantáneo 3D respecto al waypoint = 3 floats
        - Vector de deriva medio de la base 3D = 3 floats
        - Deriva escalar media de la base = 1 float
        - Desplazamiento del parche r_corner = 1 float
        # shape: (30,) o (B, 30)
        """
        canonical_kps = [
            "l_corner",
            "r_corner",
            "crotch",
            "top_ctr_f",
            "top_ctr_b",
            "l_leg_o",
            "r_leg_o",
        ]

        kp_coords = []
        for kp_name in canonical_kps:
            pos_3d = self.backend.get_keypoint_pos(kp_name)
            kp_coords.append(pos_3d)

        # Waypoint canónico instantáneo en 3D
        x_nom_single = self.traj.get_position(tau) if self.traj is not None else np.zeros(3, dtype=np.float32)

        if self.stage == 1:
            p_act = self.backend.get_keypoint_pos("l_corner")
        elif self.stage == 2:
            p_act = self.backend.get_grasp_center_pos()
        else:
            p_act = (self.backend.get_keypoint_pos("l_leg_o") + self.backend.get_keypoint_pos("r_leg_o")) / 2.0

        drift_info = self.backend.get_realtime_drift_metrics()

        if self.n_envs > 1:
            kps_flat = np.concatenate(kp_coords, axis=1).astype(np.float32) # shape: (B, 21)
            tau_val = np.full((self.n_envs, 1), np.clip(tau, 0.0, 1.0), dtype=np.float32) # shape: (B, 1)
            err_vec = (x_nom_single[None, :] - p_act).astype(np.float32) # shape: (B, 3)
            vec_drift = np.asarray(drift_info["vec_base_drift"], dtype=np.float32).reshape(self.n_envs, 3) # shape: (B, 3)
            mean_drift = np.asarray(drift_info["mean_base_drift"], dtype=np.float32).reshape(self.n_envs, 1) # shape: (B, 1)
            r_disp = np.asarray(drift_info["r_corner_disp"], dtype=np.float32).reshape(self.n_envs, 1) # shape: (B, 1)
            obs = np.concatenate([kps_flat, tau_val, err_vec, vec_drift, mean_drift, r_disp], axis=1).astype(np.float32) # shape: (B, 30)
        else:
            kps_flat = np.concatenate(kp_coords, axis=0).astype(np.float32) # shape: (21,)
            tau_val = np.array([np.clip(tau, 0.0, 1.0)], dtype=np.float32) # shape: (1,)
            err_vec = (x_nom_single - p_act).astype(np.float32) # shape: (3,)
            vec_drift = np.asarray(drift_info["vec_base_drift"], dtype=np.float32).reshape(3) # shape: (3,)
            mean_drift = np.array([float(drift_info["mean_base_drift"])], dtype=np.float32) # shape: (1,)
            r_disp = np.array([float(drift_info["r_corner_disp"])], dtype=np.float32) # shape: (1,)
            obs = np.concatenate([kps_flat, tau_val, err_vec, vec_drift, mean_drift, r_disp], axis=0).astype(np.float32) # shape: (30,)

        return obs

    def render(self):
        """
        En Genesis, el visor 3D se refresca automáticamente en cada paso de simulación
        si show_viewer=True al instanciar el entorno.
        """
        pass

    def close(self):
        """Libera recursos del entorno."""
        self.backend.release_all_keypoints()

    def start_recording(self, video_path: str, fps: float = 30.0):
        """Inicia la grabacion de video MP4 si la camara esta habilitada."""
        self.backend.start_recording(video_path, fps=fps)

    def stop_recording(self):
        """Detiene y finaliza la grabacion del video MP4."""
        self.backend.stop_recording()


if __name__ == "__main__":
    print("[*] Iniciando prueba basica de DynamicFoldingEnv...")
    # Creamos un entorno con pocos pasos para verificar que todo responda bien
    test_env = DynamicFoldingEnv(show_viewer=False, rl_horizon_steps=5)

    print("[1/3] Reiniciando entorno en Etapa 1...")
    init_obs, reset_info = test_env.reset(stage=1)
    print(f"      Observacion inicial recibida. Dimension: {init_obs.shape}")
    print(f"      Punto de inicio (P0): {reset_info['p0']}")
    print(f"      Objetivo final (P_target): {reset_info['target_pos']}")

    # Comprobacion de seguridad: vector de 30 elementos y sin valores invalidos (NaN)
    assert init_obs.shape == test_env.observation_space.shape, f"Error: la observacion debe ser {test_env.observation_space.shape}, tiene {init_obs.shape}"
    assert not np.isnan(init_obs).any(), "Error: hay valores NaN en la observacion"

    print("[2/3] Ejecutando un paso con accion cero (solo trayectoria nominal)...")
    zero_action = np.zeros(3, dtype=np.float32)
    next_o, r, term, trunc, step_inf = test_env.step(zero_action)
    print(f"      Recompensa obtenida: {r:.4f}")
    print(f"      Distancia al objetivo: {step_inf['dist_to_target']:.4f} m (Progreso tau={step_inf['tau']:.2f})")
    assert not np.isnan(next_o).any(), "Error: hay valores NaN tras el paso"

    print("[3/3] Cerrando entorno...")
    test_env.close()
    print("[OK] DynamicFoldingEnv funciona y responde correctamente!")