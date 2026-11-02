"""
test_foldnet_generalization.py
==============================
Test unitario de verificación de generalización sobre el dataset FoldNet.
Evalúa que la física, las trayectorias analíticas Bézier y el entorno Gymnasium
se adapten invariablemente a prendas de distintas geometrías (cortas, medianas, largas y anchas).
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to sys.path
WS_ROOT = Path(__file__).resolve().parent.parent
if str(WS_ROOT) not in sys.path:
    sys.path.insert(0, str(WS_ROOT))

from envs.cloth_genesis_backend import ClothGenesisBackend
from envs.dynamic_folding_env import DynamicFoldingEnv


def test_foldnet_generalization():
    print("=" * 80)
    print("  TEST DE VERIFICACION FEYNMAN: GENERALIZACION MULTI-PRENDA FOLDNET")
    print("=" * 80)

    # Conjunto de prueba representativo con gran variabilidad geométrica:
    # - 1:   Pantalón corto y ancho (L ~ 0.52m, W ~ 0.56m)
    # - 430: Pantalón de referencia nominal (L ~ 0.61m, W ~ 0.37m)
    # - 100: Pantalón largo y medio (L ~ 0.87m, W ~ 0.48m)
    test_garments = [1, 430, 100]

    for gid in test_garments:
        print(f"\n[+] Evaluando Prenda FoldNet ID: {gid}...")
        
        # 1. Instanciación del entorno Gymnasium con la prenda correspondiente
        env = DynamicFoldingEnv(
            garment_id=gid,
            show_viewer=False,
            enable_camera=False,
            rl_horizon_steps=5,
            substeps_per_action=20,
            n_envs=1,
        )
        backend = env.backend

        # 2. Verificación del backend físico y metadatos dinámicos
        assert backend.cloth_length > 0.40 and backend.cloth_length < 1.10, \
            f"Fallo en cálculo de longitud de prenda {gid}: {backend.cloth_length}"
        assert backend.cloth_width > 0.25 and backend.cloth_width < 0.75, \
            f"Fallo en cálculo de ancho de prenda {gid}: {backend.cloth_width}"
        
        print(f"    - Dimensiones extraídas: Largo = {backend.cloth_length:.3f} m, Ancho = {backend.cloth_width:.3f} m")
        print(f"    - Partículas en simulación: {backend.cloth.get_particles_pos().shape[-2]}")

        # 3. Verificación de anclaje y target de Etapa 1
        info_s1 = backend.bind_stage_keypoints(stage=1, anchor_mode="free")
        assert len(backend.active_grasp_indices) >= 5, \
            f"Fallo: agarre insuficiente en Etapa 1 para {gid} ({len(backend.active_grasp_indices)} partículas)"
        assert not np.isnan(info_s1["p_target"]).any(), "Fallo: NaNs en target de Etapa 1"
        print(f"    - Etapa 1: Agarre en {len(backend.active_grasp_indices)} partículas | Target X = {info_s1['p_target'][0]:.3f} m")

        # 4. Verificación de anclaje bicapa y parámetros de Etapa 2
        info_s2 = backend.bind_stage_keypoints(stage=2, anchor_mode="free")
        assert len(backend.active_grasp_indices) >= 10, \
            f"Fallo: pretina insuficiente en Etapa 2 para {gid} ({len(backend.active_grasp_indices)} partículas)"
        assert not np.isnan(info_s2["p0"]).any(), "Fallo: NaNs en P0 de Etapa 2"
        print(f"    - Etapa 2: Pretina bicapa en {len(backend.active_grasp_indices)} partículas | Apex Z = {info_s2['cloth_length'] + 0.13:.3f} m")

        # 5. Verificación de Gym reset y step con acción nominal
        obs, reset_info = env.reset(stage=1)
        assert obs.shape == (30,), f"Fallo dimensional en observación: {obs.shape}"
        assert not np.isnan(obs).any(), "Fallo: NaNs en vector de observación"

        # Ejecución de un paso con acción nula (nominal)
        zero_action = np.zeros(5, dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(zero_action)
        assert not np.isnan(reward), "Fallo: NaNs en recompensa"
        assert not np.isnan(next_obs).any(), "Fallo: NaNs en next_obs"

        # Verificación de que las partículas permanezcan en el workspace
        particles = env.backend.get_particles_pos()
        assert not np.isnan(particles).any(), "Fallo: NaNs en partículas físicas"
        assert np.all(particles[..., 2] >= env.backend.table_top - 0.02), "Fallo: Penetración profunda en mesa"
        assert np.all(particles[..., 2] <= 1.50), "Fallo: Partículas eyectadas hacia arriba"

        print(f"    [OK] Prenda {gid} validada exitosamente en simulación y Gym.")

    print("\n" + "=" * 80)
    print("  RESULTADO: TODOS LOS TESTS DE GENERALIZACION FOLDNET SUPERADOS (100% OK)")
    print("=" * 80)


if __name__ == "__main__":
    test_foldnet_generalization()
