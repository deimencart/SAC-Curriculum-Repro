"""
curriculum_callback.py
======================
Curriculum Learning Manager for Dynamic Cloth Folding.
"""

from typing import Dict, Any, List
import numpy as np
from pathlib import Path


STAGE1_LEVELS = [
    # r_corner_drift_tol: desplazamiento maximo tolerado del parche r_corner (m)
    # r_drift_penalty_w:  peso de la penalizacion en la funcion de recompensa
    # Niveles 1-3 (pinned): Genesis fija parche r_corner, tol laxa solo para coherencia de curriculo.
    # Nivel 4 (soft):      Fija 1 solo vertice (r_corner) como pivote; la red aprende a no arrastrar.
    # Niveles 5-8 (free):  la politica DEBE operar con CERO fijaciones en la base; umbral se endurece.
    {"level": 1, "anchor_mode": "pinned", "tau_start": 0.50, "friction": 80.0,
     "r_corner_drift_tol": 0.025, "r_drift_penalty_w": 2.0,
     "desc": "Cercano (12cm) + Base sujeta"},
    {"level": 2, "anchor_mode": "pinned", "tau_start": 0.25, "friction": 80.0,
     "r_corner_drift_tol": 0.025, "r_drift_penalty_w": 2.0,
     "desc": "Medio (25cm) + Base sujeta"},
    {"level": 3, "anchor_mode": "pinned", "tau_start": 0.00, "friction": 80.0,
     "r_corner_drift_tol": 0.025, "r_drift_penalty_w": 3.0,
     "desc": "Completo (38cm) + Base sujeta"},
    {"level": 4, "anchor_mode": "soft",   "tau_start": 0.00, "friction": 80.0,
     "r_corner_drift_tol": 0.025, "r_drift_penalty_w": 3.5,
     "desc": "Completo (38cm) + Sujecion Suave (1 Vertice Pivote)"},
    {"level": 5, "anchor_mode": "free",   "tau_start": 0.00, "friction": 80.0,
     "r_corner_drift_tol": 0.020, "r_drift_penalty_w": 4.0,
     "desc": "Completo + CERO SUJECION (Base libre, Friccion 80)"},
    {"level": 6, "anchor_mode": "free",   "tau_start": 0.00, "friction": 40.0,
     "r_corner_drift_tol": 0.015, "r_drift_penalty_w": 4.5,
     "desc": "Completo + Base libre + Friccion 40"},
    {"level": 7, "anchor_mode": "free",   "tau_start": 0.00, "friction": 15.0,
     "r_corner_drift_tol": 0.012, "r_drift_penalty_w": 5.0,
     "desc": "Completo + Base libre + Friccion 15"},
    {"level": 8, "anchor_mode": "free",   "tau_start": 0.00, "friction": 5.0,
     "r_corner_drift_tol": 0.008, "r_drift_penalty_w": 6.0,
     "desc": "Completo + Base libre + Friccion real (5.0)"},
]

STAGE2_LEVELS = [
    # Currículo Inverso por Poses Clave para Etapa 2 (Estiramiento Longitudinal y Alineación de Tobillos):
    # Nivel 1: Pose C (Frontal asentado, tobillos en mesa, tau_start=0.65, H=20, sub=60, fricción 80.0)
    #          Dominio del arrastre cuasi-estatico y traccion sobre la mesa sin deslizamiento de tobillos.
    {
        "level": 1,
        "anchor_mode": "free",
        "tau_start": 0.65,
        "rl_horizon": 20,
        "substeps": 60,
        "friction": 80.0,
        "ankle_tol": 0.045,
        "checkpoint": "pantalon_430_stage2_pose_c_grounded.pt",
        "desc": "Nivel 1 (Pose C): Traccion cuasi-estatica en mesa (tau 0.65->1.00, H=20, sub=60, mu=80)",
    },
    # Nivel 2: Pose B (Aéreo asentado, tobillos acoplados a ~3.5cm, tau_start=0.35, H=20, sub=60, fricción 80.0)
    #          Dominio del avance longitudinal y posado gravitatorio con tobillos acoplados.
    {
        "level": 2,
        "anchor_mode": "free",
        "tau_start": 0.35,
        "rl_horizon": 20,
        "substeps": 60,
        "friction": 80.0,
        "ankle_tol": 0.045,
        "checkpoint": "pantalon_430_stage2_pose_b_airborne.pt",
        "desc": "Nivel 2 (Pose B): Avance aéreo + Posado gravitatorio + Tracción (tau 0.35->1.00, H=20, sub=60, mu=80)",
    },
    # Nivel 3: Pose A (Reposo tras Etapa 1, ciclo completo, tau_start=0.00, H=20, sub=60, fricción 80.0)
    #          Ciclo completo multi-fase: Elevación + Reposo aéreo + Avance + Reposo frontal + Tracción.
    {
        "level": 3,
        "anchor_mode": "free",
        "tau_start": 0.00,
        "rl_horizon": 20,
        "substeps": 60,
        "friction": 80.0,
        "ankle_tol": 0.045,
        "checkpoint": "pantalon_430_folded_stage1.pt",
        "desc": "Nivel 3 (Pose A): Ciclo Completo Multi-Fase (tau 0.00->1.00, H=20, sub=60, mu=80)",
    },
    # Nivel 4: Pose A (Ciclo completo, fricción reducida a 15.0)
    {
        "level": 4,
        "anchor_mode": "free",
        "tau_start": 0.00,
        "rl_horizon": 20,
        "substeps": 60,
        "friction": 15.0,
        "ankle_tol": 0.045,
        "checkpoint": "pantalon_430_folded_stage1.pt",
        "desc": "Nivel 4 (Pose A): Ciclo Completo + Fricción reducida (15.0)",
    },
    # Nivel 5: Pose A (Ciclo completo, fricción nominal real a 5.0)
    {
        "level": 5,
        "anchor_mode": "free",
        "tau_start": 0.00,
        "rl_horizon": 20,
        "substeps": 60,
        "friction": 5.0,
        "ankle_tol": 0.045,
        "checkpoint": "pantalon_430_folded_stage1.pt",
        "desc": "Nivel 5 (Pose A): Ciclo Completo + Fricción Real Nominal (5.0)",
    },
]


class CurriculumManager:
    def __init__(
        self,
        target_success_rate: float = 0.85, # 85% de aprobacion para promocion
        window_size: int = 15,              # ventana movil de ultimos intentos
        friction_min: float = 5.0,          # Friccion real mundo fisico
        friction_max: float = 80.0,         # Friccion maxima inicial
        max_stage: int = 4,                 # Soporte para hasta 4 etapas
        start_level: int = 1,               # Nivel curricular inicial
        checkpoints_dir: str = "D:/genesis_ws/checkpoints/Canonicals",  
    ):
        self.target_success_rate = target_success_rate
        self.window_size = window_size
        self.friction_min = friction_min
        self.friction_max = friction_max
        self.max_stage = max_stage
        self.checkpoints_dir = Path(checkpoints_dir)

        self.history: List[bool] = []
        self.current_stage = 1
        self._update_levels_for_stage()
        self.current_level_idx = max(0, min(len(self.levels) - 1, start_level - 1))

    def _update_levels_for_stage(self):
        """Asigna la tabla de niveles curriculares correspondiente a la etapa actual."""
        if self.current_stage == 1:
            self.levels = STAGE1_LEVELS
        elif self.current_stage == 2:
            self.levels = STAGE2_LEVELS
        else:
            self.levels = STAGE1_LEVELS

    def set_stage(self, stage: int, start_level: int = 1):
        """Configura explícitamente la etapa y nivel inicial."""
        self.current_stage = int(stage)
        self._update_levels_for_stage()
        self.current_level_idx = max(0, min(len(self.levels) - 1, start_level - 1))
        self.history.clear()

    @property
    def current_config(self) -> Dict[str, Any]:
        return self.levels[self.current_level_idx]

    @property
    def current_friction(self) -> float:
        return float(self.current_config["friction"])

    def get_env_options(self) -> Dict[str, Any]:
        """Devuelve las opciones para env.reset(options=...)."""
        cfg = self.current_config
        if "checkpoint" in cfg:
            ckpt = str(self.checkpoints_dir / cfg["checkpoint"])
        else:
            ckpt = self.get_checkpoint_for_stage(self.current_stage)
        opts = {
            "anchor_mode": cfg.get("anchor_mode", "free"),
            "tau_start": cfg.get("tau_start", 0.0),
            "friction": cfg.get("friction", 80.0),
            "curriculum_level": cfg["level"],
            "checkpoint_path": ckpt if ckpt else None,
            "rl_horizon": cfg.get("rl_horizon", 20),
            "substeps": cfg.get("substeps", 60 if self.current_stage == 2 else 35),
            "ankle_tol": cfg.get("ankle_tol", 0.045),
            # Politica de restriccion de desplazamiento de r_corner (se endurece con el nivel)
            "r_corner_drift_tol": cfg.get("r_corner_drift_tol", 0.025),
            "r_drift_penalty_w": cfg.get("r_drift_penalty_w", 2.0),
        }
        return opts

    def record_episode(self, stage: int, episode_return: float, success: bool) -> Dict[str, Any]:

        """
        Anota el resultado del intento actual y decide si sube de nivel curricular o avanza de etapa.
        """
        self.history.append(bool(success))
        if len(self.history) > self.window_size:
            self.history.pop(0)

        aciertos = sum(self.history)
        total = len(self.history)
        success_rate = float(aciertos / total) if total > 0 else 0.0
        should_promote = False
        next_stage = self.current_stage
        promoted_level = False

        if total >= 10 and success_rate >= self.target_success_rate:
            # Si aún quedan niveles en la etapa actual, avanzamos de nivel
            if self.current_level_idx < len(self.levels) - 1:
                self.current_level_idx += 1
                promoted_level = True
                self.history.clear()
            else:
                # Si ya dominó todos los niveles de la etapa, avanza de etapa si max_stage lo permite
                if self.current_stage < self.max_stage:
                    should_promote = True
                    self.current_stage += 1
                    next_stage = self.current_stage
                    self._update_levels_for_stage()
                    self.current_level_idx = 0
                    self.history.clear()

        cfg = self.current_config
        if "checkpoint" in cfg:
            ckpt_to_load = str(self.checkpoints_dir / cfg["checkpoint"])
        else:
            ckpt_to_load = self.get_checkpoint_for_stage(next_stage)

        return {
            "stage": self.current_stage,
            "next_stage": next_stage,
            "level": cfg["level"],
            "level_desc": cfg["desc"],
            "anchor_mode": cfg["anchor_mode"],
            "tau_start": cfg["tau_start"],
            "rl_horizon": cfg.get("rl_horizon", 20),
            "success_rate": success_rate,
            "table_friction": cfg["friction"],
            "friction": cfg["friction"],
            "should_promote": should_promote,
            "promoted_level": promoted_level,
            "checkpoint_to_load": ckpt_to_load,
        }
    def get_checkpoint_for_stage(self, stage: int) -> str:
        """Devuelve la pose canonica de la que debe arrancar la etapa."""
        if stage == 2:
            return str(self.checkpoints_dir / "pantalon_430_folded_stage1.pt")
        elif stage == 3:
            return str(self.checkpoints_dir / "pantalon_430_folded_stage2.pt")
        elif stage >= 4:
            return str(self.checkpoints_dir / "pantalon_430_folded_stage3.pt")
        return "" # La Etapa 1 arranca con la prenda extendida original

if __name__ == "__main__":
    print("[*] Probando CurriculumManager...")
    manager = CurriculumManager(target_success_rate=0.85, window_size=15)

    print("\n--- Simulando 8 episodios con fallos (el robot está aprendiendo) ---")
    for ep in range(1, 9):
        estado = manager.record_episode(stage=1, episode_return=-1.0, success=False)
        print(f"Episodio {ep}: Aciertos={estado['success_rate']*100:.1f}%, Friccion={estado['table_friction']}, Promover={estado['should_promote']}")

    print("\n--- Simulando 12 episodios exitosos consecutivos ---")
    for ep in range(9, 21):
        estado = manager.record_episode(stage=1, episode_return=1.0, success=True)
        print(f"Episodio {ep}: Aciertos={estado['success_rate']*100:.1f}%, Friccion={estado['table_friction']}, Promover={estado['should_promote']}")
        if estado['should_promote']:
            print(f"--> ¡GRADUADO A ETAPA {estado['next_stage']}! Checkpoint a cargar: {estado['checkpoint_to_load']}")
            break

    print("[OK] CurriculumManager probado y funcionando!")