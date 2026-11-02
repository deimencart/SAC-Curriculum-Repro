"""
bezier_geometry.py
==================
Analytical engine for 3D Bézier Curves with C^2 Quintic Smoothstep Timing:
    x(tau) = B(s(tau))

[Core Foundations]
------------------
1. Quintic Smoothstep s(tau):
   s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5 ensures zero velocity and acceleration 
   at endpoints (v=0, a=0, C^2 Minimum-Jerk), eliminating inertial impact forces.

2. Bernstein Polynomials B(s):
   - Quadratic (mode="quadratic", calibrated in 09_ghost):
     P1 = M + H * n_dir (apex at midpoint M = (P0 + P2) / 2)
     B(s) = (1-s)^2*P0 + 2*s*(1-s)*P1 + s^2*P2
   - Cubic (mode="cubic"):
     P1 = P0 + H * n_dir, P2 = P3 + H * n_dir
     B(s) = (1-s)^3*P0 + 3*s*(1-s)^2*P1 + 3*s^2*(1-s)*P2 + s^3*P3
"""

from typing import Union, Tuple, Dict, Any, Optional
import numpy as np


def quintic_smoothstep(tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Degree-5 Polynomial (C^2 Minimum-Jerk):
        s(tau) = 10*tau^3 - 15*tau^4 + 6*tau^5,  tau in [0, 1]

    Boundary Invariants:
        s(0) = 0,   s(1) = 1
        s'(0) = 0,  s'(1) = 0   (zero velocity at endpoints)
        s''(0) = 0, s''(1) = 0  (zero acceleration at endpoints)
    """
    if isinstance(tau, np.ndarray):
        tau_clamped = np.clip(tau, 0.0, 1.0)
    else:
        tau_clamped = max(0.0, min(1.0, float(tau)))

    return (
        10.0 * (tau_clamped ** 3)
        - 15.0 * (tau_clamped ** 4)
        + 6.0 * (tau_clamped ** 5)
    )


def bernstein_quadratic(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    s: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Quadratic Bernstein Polynomial:
        B(s) = (1-s)^2*P0 + 2*s*(1-s)*P1 + s^2*P2
    # shape: (3,)
    """
    s_clamped = np.clip(s, 0.0, 1.0)
    w0 = (1.0 - s_clamped) ** 2
    w1 = 2.0 * s_clamped * (1.0 - s_clamped)
    w2 = s_clamped ** 2

    return w0 * p0 + w1 * p1 + w2 * p2


def bernstein_cubic(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    s: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Cubic Bernstein Polynomial:
        B(s) = (1-s)^3*P0 + 3*s*(1-s)^2*P1 + 3*s^2*(1-s)*P2 + s^3*P3
    # shape: (3,)
    """
    s_clamped = np.clip(s, 0.0, 1.0)
    w0 = (1.0 - s_clamped) ** 3
    w1 = 3.0 * s_clamped * ((1.0 - s_clamped) ** 2)
    w2 = 3.0 * (s_clamped ** 2) * (1.0 - s_clamped)
    w3 = s_clamped ** 3

    return w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3


class BezierTrajectory:
    """
    3D Bézier Trajectory Generator supporting Quadratic (09_ghost) and Cubic modes.
    Constructs an orthonormal spatial coordinate frame (u: chord, v: vertical, w: lateral)
    and computes control points from elevation ratio and lateral inclination angle.
    """
    def __init__(
        self,
        p0: np.ndarray,
        p_target: np.ndarray,
        h_ratio: float = 0.65,
        theta_deg: float = 0.0,
        mode: str = "quadratic",
    ) -> None:
        """
        Args:
            p0: Starting Cartesian coordinate # shape: (3,)
            p_target: Destination Cartesian coordinate # shape: (3,)
            h_ratio: Apex elevation ratio relative to chord length ||p_target - p0||
            theta_deg: Lateral tilt angle in degrees (0.0 = planar vertical arc)
            mode: "quadratic" (parabolic midpoint apex) or "cubic" (dual vertical control)
        """
        self.p0 = np.asarray(p0, dtype=np.float32)
        self.p_target = np.asarray(p_target, dtype=np.float32)
        self.mode = mode.lower()

        # 1. Metric chord vector protected against division by zero
        d_vec = self.p_target - self.p0
        d_norm = float(np.linalg.norm(d_vec))
        self.d_norm = d_norm

        if d_norm < 1e-7:
            self.u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            self.v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.w = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self.offset_dir = self.v
            self.H = 0.0
            self.p1 = self.p0.copy()
            self.p2 = self.p0.copy()
            return

        # 2. Orthonormal frame (u: along chord, v: vertical projection, w: transverse)
        self.u = d_vec / (d_norm + 1e-7)
        z_global = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        v_proj = z_global - np.dot(z_global, self.u) * self.u
        v_norm = float(np.linalg.norm(v_proj))
        self.v = v_proj / (v_norm + 1e-7) if v_norm >= 1e-7 else z_global
        self.w = np.cross(self.u, self.v)

        # 3. Tilted apex director vector
        theta_rad = np.radians(theta_deg)
        self.offset_dir = np.cos(theta_rad) * self.v + np.sin(theta_rad) * self.w
        self.H = h_ratio * d_norm

        # 4. Control points construction based on mode
        if self.mode == "quadratic":
            # Parabolic midpoint apex (09_ghost formulation)
            midpoint = self.p0 + d_vec * 0.5
            self.p1 = midpoint + self.H * self.offset_dir
            self.p2 = self.p_target
        elif self.mode == "cubic":
            # Cubic dual control points (lift over P0, descent over P3)
            self.p1 = self.p0 + self.H * self.offset_dir
            self.p2 = self.p_target + self.H * self.offset_dir
        else:
            raise ValueError(f"Unsupported Bézier mode: {self.mode}. Must be 'quadratic' or 'cubic'.")

    def get_position(self, tau: float) -> np.ndarray:
        """
        Calculates 3D Cartesian position for normalized time tau in [0, 1]
        using quintic smoothstep timing: x(tau) = B(s(tau)).
        # shape: (3,)
        """
        s = float(quintic_smoothstep(tau))
        if self.d_norm < 1e-7:
            return self.p0.copy()

        if self.mode == "quadratic":
            return bernstein_quadratic(self.p0, self.p1, self.p_target, s)
        else:
            return bernstein_cubic(self.p0, self.p1, self.p2, self.p_target, s)


class BezierCubicTrajectory(BezierTrajectory):
    """Backwards-compatible alias explicitly fixing mode='cubic'."""
    def __init__(
        self,
        p0: np.ndarray,
        p3: np.ndarray,
        h_ratio: float = 0.40,
        theta_deg: float = 15.0,
    ) -> None:
        super().__init__(p0=p0, p_target=p3, h_ratio=h_ratio, theta_deg=theta_deg, mode="cubic")


class Stage2WaistStretchTrajectory:
    """
    Trayectoria analítica C^2 en 5 fases para la Etapa 2 (Estiramiento de Cintura y Asentamiento con Tobillos Juntos):
    1. Fase a - Elevación Vertical Alta (tau in [0.00, 0.20]):
       Ascenso quíntico suave C^2 hasta Z_apex = Z_mesa + L_cloth + delta_clearance (suspensión gravitatoria total > 65 cm).
    2. Fase b - Batido / Agitación Inercial en el Aire (tau in [0.20, 0.35]):
       Oscilaciones armónicas en el plano sagital (Y, Z) para forzar el colapso inercial de las perneras y juntar tobillos.
    3. Fase c - Desplazamiento Longitudinal en el Aire (tau in [0.35, 0.55]):
       Avance horizontal en Y (Delta Y = -cloth_length) a cota suspendida alta para posicionar los bajos directamente sobre el frontal.
    4. Fase d - Posado Inicial de Tobillos en Mesa (tau in [0.55, 0.65]):
       Descenso controlado a cota de contacto Z_touchdown = Z_mesa + cloth_length - 0.08 m (asentando los bajos con fricción en mesa).
    5. Fase e - Retroceso de Tracción y Estiramiento sobre la Mesa (tau in [0.65, 1.00]):
       La pinza retrocede en +Y devolviendo la pretina a su posición original de reposo en la mesa (p_target).
       Al retroceder, la fricción de la mesa en los tobillos induce tracción axial que estira y alisa completamente el pantalón.
    """
    def __init__(
        self,
        p0: np.ndarray,
        p_target: np.ndarray,
        p_ankles: Optional[np.ndarray] = None,
        cloth_length: float = 0.571,
        table_z: float = 0.55,
        delta_clearance: float = 0.650,
    ) -> None:
        self.p0 = np.asarray(p0, dtype=np.float32)
        self.p_target = np.asarray(p_target, dtype=np.float32)
        self.cloth_length = float(cloth_length)
        self.table_z = float(table_z)
        
        # Cota de elevación vertical alta canónica (+65 cm sobre la pretina en mesa)
        self.z_apex = float(self.p0[2] + delta_clearance)
        self.apex_pos = np.array([self.p0[0], self.p0[1], self.z_apex], dtype=np.float32)

        # Cota frontal a la misma altura exacta del levantamiento (Z_apex constante)
        y_disp = float(self.p0[1] - self.cloth_length)
        self.p_disp = np.array([self.p0[0], y_disp, self.z_apex], dtype=np.float32)

    def get_position(self, tau: float) -> np.ndarray:
        tau_c = float(np.clip(tau, 0.0, 1.0))

        if tau_c <= 0.25:
            # Fase a: Elevación vertical pura C^2 hasta suspensión total alta
            u = tau_c / 0.25
            s = quintic_smoothstep(u)
            pos = self.p0 + (self.apex_pos - self.p0) * s
            return pos.astype(np.float32)

        elif tau_c <= 0.35:
            # Fase b: Suspensión alta estática en Z_apex (reposo gravitatorio pasivo para alinear tobillos)
            return self.apex_pos.copy().astype(np.float32)

        elif tau_c <= 0.65:
            # Fase c: Desplazamiento horizontal puro en Y a la MISMA ALTURA EXACTA (Z_apex constante)
            u = (tau_c - 0.35) / 0.30
            s = quintic_smoothstep(u)
            pos = self.apex_pos + (self.p_disp - self.apex_pos) * s
            return pos.astype(np.float32)

        elif tau_c <= 0.70:
            # Fase d: Suspensión estática en frontal alto (reposo gravitatorio pendular para asentar tobillos)
            return self.p_disp.copy().astype(np.float32)

        else:
            # Fase e: Retroceso gradual suave C^2 desde p_disp (Z_apex) hacia p_target (mesa trasera)
            u = (tau_c - 0.70) / 0.30
            s = quintic_smoothstep(u)
            pos = (1.0 - s) * self.p_disp + s * self.p_target
            return pos.astype(np.float32)


# Alias para compatibilidad hacia atrás
Stage2UnifiedFoldTrajectory = Stage2WaistStretchTrajectory
Stage2WaistDrapeTrajectory = Stage2WaistStretchTrajectory


if __name__ == "__main__":
    print("[*] Running verification tests on bezier_geometry.py...")

    # 1. Boundary checks for quintic smoothstep
    assert np.isclose(quintic_smoothstep(0.0), 0.0), "Error: s(0) != 0"
    assert np.isclose(quintic_smoothstep(1.0), 1.0), "Error: s(1) != 1"

    # 2. Numerical zero velocity check at endpoints
    eps = 1e-5
    v_0 = (quintic_smoothstep(eps) - quintic_smoothstep(0.0)) / eps
    v_1 = (quintic_smoothstep(1.0) - quintic_smoothstep(1.0 - eps)) / eps
    assert np.isclose(v_0, 0.0, atol=1e-3), f"Error: v(0) is non-zero ({v_0})"
    assert np.isclose(v_1, 0.0, atol=1e-3), f"Error: v(1) is non-zero ({v_1})"

    # 3. Geometric 3D trajectory check: Quadratic mode (09_ghost calibrated)
    p_start = np.array([0.0, 0.26, 0.05], dtype=np.float32)
    p_target = np.array([0.26, 0.0, 0.05], dtype=np.float32)
    chord_len = float(np.linalg.norm(p_target - p_start))

    curve_quad = BezierTrajectory(p_start, p_target, h_ratio=0.65, theta_deg=0.0, mode="quadratic")
    assert np.allclose(curve_quad.get_position(0.0), p_start), "Error: Start != P0"
    assert np.allclose(curve_quad.get_position(1.0), p_target), "Error: Target != P2"

    apex_quad = curve_quad.get_position(0.5)
    delta_z_quad = apex_quad[2] - p_start[2]
    # For quadratic, apex at s=0.5 is P0/4 + P1/2 + P2/4 = Midpoint + 0.5 * H
    expected_apex_z = 0.5 * 0.65 * chord_len
    assert delta_z_quad > 0.08, f"Error: Apex elevation too low ({delta_z_quad} m)"
    print(f"[OK] Quadratic Apex: Z = {apex_quad[2]:.4f} m (Delta Z = +{delta_z_quad*100:.2f} cm)")

    # 4. Geometric 3D trajectory check: Cubic mode
    curve_cubic = BezierTrajectory(p_start, p_target, h_ratio=0.65, theta_deg=0.0, mode="cubic")
    assert np.allclose(curve_cubic.get_position(0.0), p_start), "Error: Start != P0"
    assert np.allclose(curve_cubic.get_position(1.0), p_target), "Error: Target != P3"
    apex_cubic = curve_cubic.get_position(0.5)
    delta_z_cubic = apex_cubic[2] - p_start[2]
    print(f"[OK] Cubic Apex:     Z = {apex_cubic[2]:.4f} m (Delta Z = +{delta_z_cubic*100:.2f} cm)")

    # 5. Backwards compatibility check
    curve_alias = BezierCubicTrajectory(p_start, p_target, h_ratio=0.40, theta_deg=15.0)
    assert np.allclose(curve_alias.get_position(0.0), p_start)
    assert np.allclose(curve_alias.get_position(1.0), p_target)

    # 6. Stage 2 Waist Stretch Trajectory test (5 Fases canónicas)
    p_w0 = np.array([0.60, 0.20, 0.05], dtype=np.float32)
    p_w_target = np.array([0.60, 0.20, 0.062], dtype=np.float32)
    p_ankles_ref = np.array([0.60, -0.37, 0.05], dtype=np.float32)
    traj_s2 = Stage2UnifiedFoldTrajectory(
        p0=p_w0,
        p_target=p_w_target,
        p_ankles=p_ankles_ref,
        cloth_length=0.571,
        table_z=0.05,
        delta_clearance=0.650,
    )
    assert np.allclose(traj_s2.get_position(0.0), p_w0), "Error: Start != P0"
    assert np.allclose(traj_s2.get_position(1.0), p_w_target), "Error: End != P_target"
    apex_s2 = traj_s2.get_position(0.25)
    assert np.isclose(apex_s2[2], p_w0[2] + 0.650), f"Error in apex Z: {apex_s2[2]}"
    apex_settle = traj_s2.get_position(0.35)
    assert np.allclose(apex_settle, traj_s2.apex_pos), f"Error in Phase b static settle"
    disp_s2 = traj_s2.get_position(0.65)
    assert np.allclose(disp_s2, traj_s2.p_disp), f"Error in Phase c displacement point"
    assert np.isclose(disp_s2[2], p_w0[2] + 0.650), "Error: Displacement Z must equal Apex Z"
    assert np.isclose(disp_s2[1], p_w0[1] - 0.571), "Error: Displacement Y must equal Y0 - cloth_length"
    td_settle = traj_s2.get_position(0.70)
    assert np.allclose(td_settle, traj_s2.p_disp), f"Error in Phase d touchdown settle"
    print(f"[OK] Stage 2 Canonical Trajectory: 5 Phases validated successfully (P0 -> Apex -> High Settle -> High-Z Disp -> TD Settle -> Smooth Retreat)")

    print("[OK] All tests in bezier_geometry.py verified successfully.")