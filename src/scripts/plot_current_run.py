import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil
from pathlib import Path

jsonl_path = Path(r"D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_metrics.jsonl")
lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
run = [l for l in lines if "curriculum_level" in l]

episodes = np.array([int(r["episode"]) for r in run])
final_dist = np.array([float(r["final_dist_m"]) * 100 for r in run]) # cm
rewards = np.array([float(r["ep_reward"]) for r in run])
r_disp = np.array([float(r.get("r_corner_mean_disp", 0.0)) * 100 for r in run]) # cm
contact_ratio = np.array([float(r.get("contact_ratio", 0.0)) for r in run])
alpha = np.array([float(r["alpha"]) for r in run])
critic_loss = np.array([float(r["critic_loss"]) for r in run])
actor_loss = np.array([float(r["actor_loss"]) for r in run])

def moving_avg(arr, w=10):
    if len(arr) < w:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad = np.full(w - 1, ma[0])
    return np.concatenate((pad, ma))

plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
fig.suptitle(f'Telemetria de Entrenamiento Actual (Etapa 1: Plegado de Pernera, Eps 1-{len(run)})', fontsize=16, fontweight='bold', color='#38bdf8')

# Panel 1: Distancia Final Cartesiana (cm)
ax1 = axs[0, 0]
ax1.scatter(episodes, final_dist, color='#60a5fa', alpha=0.45, s=20, label='Distancia real')
ax1.plot(episodes, moving_avg(final_dist, 10), color='#38bdf8', linewidth=2.5, label='Media Móvil (w=10)')
ax1.axhline(4.0, color='#22c55e', linestyle='--', linewidth=1.5, label='Umbral Meta (4.0 cm)')
ax1.axhline(5.0, color='#eab308', linestyle=':', linewidth=1.2, label='Tolerancia (5.0 cm)')
ax1.set_ylim(0, 10)
ax1.set_title('1. Error Cartesiano de Dobladillos (l_corner -> r_corner)', fontsize=12, fontweight='semibold')
ax1.set_xlabel('Episodio')
ax1.set_ylabel('Distancia Final (cm)')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)

# Panel 2: Deriva del Parche R_Corner (cm)
ax2 = axs[0, 1]
# Filtrar explosiones numéricas extremas para visualizar la banda de trabajo real
r_disp_clipped = np.clip(r_disp, 0, 15)
ax2.scatter(episodes, r_disp_clipped, color='#f87171', alpha=0.45, s=20, label='Deriva Base (acotada a 15cm)')
ax2.plot(episodes, moving_avg(r_disp_clipped, 10), color='#ef4444', linewidth=2.5, label='Media Móvil (w=10)')
ax2.axhline(2.0, color='#22c55e', linestyle='--', linewidth=1.5, label='Umbral Invarianza (2.0 cm)')
ax2.set_ylim(0, 10)
ax2.set_title('2. Deriva Física de la Pernera Base (r_corner patch)', fontsize=12, fontweight='semibold')
ax2.set_xlabel('Episodio')
ax2.set_ylabel('Desplazamiento al Asentar (cm)')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

# Panel 3: Recompensa y Temperatura Alpha (SAC)
ax3 = axs[1, 0]
color_rew = '#34d399'
ax3.scatter(episodes, rewards, color='#10b981', alpha=0.35, s=15)
ax3.plot(episodes, moving_avg(rewards, 10), color=color_rew, linewidth=2.0, label='Recompensa Episodio')
ax3.set_xlabel('Episodio')
ax3.set_ylabel('Retorno Acumulado', color=color_rew)
ax3.tick_params(axis='y', labelcolor=color_rew)
ax3.grid(True, linestyle='--', alpha=0.3)

ax3_twin = ax3.twinx()
color_alpha = '#fbbf24'
ax3_twin.plot(episodes, alpha, color=color_alpha, linewidth=2.0, linestyle='--', label='Alpha (Temperatura)')
ax3_twin.set_ylabel('Temperatura Entropía Alpha', color=color_alpha)
ax3_twin.tick_params(axis='y', labelcolor=color_alpha)
ax3.set_title('3. Retorno RL y Descenso de Temperatura Alpha', fontsize=12, fontweight='semibold')

# Panel 4: Ratio de Contacto Bicapa con Mesa
ax4 = axs[1, 1]
ax4.scatter(episodes, contact_ratio * 100, color='#a78bfa', alpha=0.45, s=20, label='Ratio Contacto Mesa (%)')
ax4.plot(episodes, moving_avg(contact_ratio * 100, 10), color='#8b5cf6', linewidth=2.5, label='Media Móvil (w=10)')
ax4.axhline(65.0, color='#22c55e', linestyle='--', linewidth=1.5, label='Límite Bicapa (<= 65% contacto)')
ax4.set_ylim(40, 80)
ax4.set_title('4. Asentamiento Bicapa (Contacto Tela-Mesa)', fontsize=12, fontweight='semibold')
ax4.set_xlabel('Episodio')
ax4.set_ylabel('% Partículas en Mesa')
ax4.grid(True, linestyle='--', alpha=0.3)
ax4.legend(loc='upper right', fontsize=9)

plt.tight_layout()

# Guardar en logs
out_logs = Path(r"D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\current_run_progress.png")
plt.savefig(out_logs, dpi=150, bbox_inches='tight')
print(f"Guardado en: {out_logs}")

# Copiar a directorio de artifacts
brain_dir = Path(r"C:\Users\lam70\.gemini\antigravity\brain\56beb8ea-a685-4296-aba0-d1fe9dcd541c")
if brain_dir.exists():
    out_brain = brain_dir / "current_run_progress.png"
    shutil.copy(out_logs, out_brain)
    print(f"Copiado a artifact dir: {out_brain}")

plt.close()
