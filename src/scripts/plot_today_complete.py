"""
plot_today_complete.py
======================
Genera el panel completo de 6 graficas de telemetria para el entrenamiento de hoy (Eps 1-200).
Exporta a logs/ y al directorio de artifacts de Antigravity.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil
from pathlib import Path

jsonl_path = Path(r"D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_metrics.jsonl")
lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

# Seleccionar los 250 episodios de la corrida nocturna (Nivel 2)
run = lines[-250:]

episodes = np.arange(1, len(run) + 1)
final_dist = np.array([float(r["final_dist_m"]) * 100 for r in run]) # cm
rewards = np.array([float(r["ep_reward"]) for r in run])
r_disp = np.array([float(r.get("base_drift", r.get("r_corner_mean_disp", 0.0))) * 100 for r in run]) # cm
contact_ratio = np.array([float(r.get("contact_ratio", 0.0)) * 100 for r in run]) # %
alpha = np.array([float(r.get("alpha", 1.0)) for r in run])
critic_loss = np.array([float(r.get("critic_loss", 0.0)) for r in run])
actor_loss = np.array([float(r.get("actor_loss", 0.0)) for r in run])

def moving_avg(arr, w=10):
    if len(arr) < w:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad = np.full(w - 1, ma[0])
    return np.concatenate((pad, ma))

plt.style.use('dark_background')
fig, axs = plt.subplots(3, 2, figsize=(16, 14), dpi=150)
fig.suptitle(f'Telemetría Completa de Entrenamiento — 250 Episodios (Nivel 2: Tau=0.25, Plegado Dinámico SAC)', 
             fontsize=17, fontweight='bold', color='#38bdf8', y=0.995)

# Panel 1: Distancia Final Cartesiana (cm)
ax1 = axs[0, 0]
ax1.scatter(episodes, final_dist, color='#60a5fa', alpha=0.45, s=22, label='Distancia Real')
ax1.plot(episodes, moving_avg(final_dist, 10), color='#38bdf8', linewidth=2.5, label='Media Móvil (w=10)')
ax1.axhline(4.0, color='#22c55e', linestyle='--', linewidth=1.5, label='Umbral Éxito (4.0 cm)')
ax1.axhline(5.0, color='#eab308', linestyle=':', linewidth=1.2, label='Tolerancia (5.0 cm)')
ax1.set_ylim(0, 10)
ax1.set_title('1. Error Cartesiano de Dobladillo (l_corner -> r_corner)', fontsize=12, fontweight='semibold')
ax1.set_xlabel('Episodio')
ax1.set_ylabel('Distancia Final (cm)')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)

# Panel 2: Deriva del Parche R_Corner (cm)
ax2 = axs[0, 1]
r_disp_clipped = np.clip(r_disp, 0, 15)
ax2.scatter(episodes, r_disp_clipped, color='#f87171', alpha=0.45, s=22, label='Deriva Base Real')
ax2.plot(episodes, moving_avg(r_disp_clipped, 10), color='#ef4444', linewidth=2.5, label='Media Móvil (w=10)')
ax2.axhline(2.0, color='#22c55e', linestyle='--', linewidth=1.5, label='Umbral Invarianza (2.0 cm)')
ax2.set_ylim(0, 10)
ax2.set_title('2. Deriva Física de la Pernera Base (r_corner patch)', fontsize=12, fontweight='semibold')
ax2.set_xlabel('Episodio')
ax2.set_ylabel('Desplazamiento al Asentar (cm)')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

# Panel 3: Retorno del Episodio
ax3 = axs[1, 0]
ax3.scatter(episodes, rewards, color='#10b981', alpha=0.4, s=20, label='Retorno')
ax3.plot(episodes, moving_avg(rewards, 10), color='#34d399', linewidth=2.2, label='Media Móvil (w=10)')
ax3.set_title('3. Retorno Acumulado por Episodio (Reward)', fontsize=12, fontweight='semibold')
ax3.set_xlabel('Episodio')
ax3.set_ylabel('Recompensa Total')
ax3.grid(True, linestyle='--', alpha=0.3)
ax3.legend(loc='lower right', fontsize=9)

# Panel 4: Asentamiento Bicapa (% Particulas en Mesa)
ax4 = axs[1, 1]
ax4.scatter(episodes, contact_ratio, color='#a78bfa', alpha=0.45, s=22, label='% Contacto Mesa')
ax4.plot(episodes, moving_avg(contact_ratio, 10), color='#8b5cf6', linewidth=2.5, label='Media Móvil (w=10)')
ax4.axhline(65.0, color='#22c55e', linestyle='--', linewidth=1.5, label='Cota Bicapa (<= 65%)')
ax4.set_ylim(40, 80)
ax4.set_title('4. Asentamiento Bicapa de la Tela (Contacto con Mesa)', fontsize=12, fontweight='semibold')
ax4.set_xlabel('Episodio')
ax4.set_ylabel('% Partículas en Contacto')
ax4.grid(True, linestyle='--', alpha=0.3)
ax4.legend(loc='upper right', fontsize=9)

# Panel 5: Perdida de los 2 Criticos Gemelos (TD MSE)
ax5 = axs[2, 0]
critic_loss_clipped = np.clip(critic_loss, 0, 12)
ax5.scatter(episodes, critic_loss_clipped, color='#fb923c', alpha=0.4, s=20, label='Critic Loss')
ax5.plot(episodes, moving_avg(critic_loss_clipped, 10), color='#f97316', linewidth=2.2, label='Media Móvil (w=10)')
ax5.set_title('5. Pérdida de Bellman de Críticos Gemelos (MSE Q1/Q2)', fontsize=12, fontweight='semibold')
ax5.set_xlabel('Episodio')
ax5.set_ylabel('Loss MSE')
ax5.grid(True, linestyle='--', alpha=0.3)
ax5.legend(loc='upper right', fontsize=9)

# Panel 6: Perdida del Actor y Temperatura Alpha
ax6 = axs[2, 1]
color_act = '#38bdf8'
ax6.scatter(episodes, actor_loss, color='#0284c7', alpha=0.35, s=18)
ax6.plot(episodes, moving_avg(actor_loss, 10), color=color_act, linewidth=2.2, label='Actor Loss (J_pi)')
ax6.set_xlabel('Episodio')
ax6.set_ylabel('Pérdida del Actor (Más negativo = Mayor Q)', color=color_act)
ax6.tick_params(axis='y', labelcolor=color_act)
ax6.grid(True, linestyle='--', alpha=0.3)

ax6_twin = ax6.twinx()
color_alp = '#fbbf24'
ax6_twin.plot(episodes, alpha, color=color_alp, linewidth=2.2, linestyle='--', label='Alpha (Temperatura)')
ax6_twin.set_ylabel('Temperatura de Entropía Alpha', color=color_alp)
ax6_twin.tick_params(axis='y', labelcolor=color_alp)
ax6.set_title('6. Actor Loss y Descenso Monótono de Temperatura Alpha', fontsize=12, fontweight='semibold')

plt.tight_layout()

# Guardar en logs
out_logs = Path(r"D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_run_today.png")
plt.savefig(out_logs, dpi=150, bbox_inches='tight')
print(f"Guardado en: {out_logs}")

# Copiar a directorio de artifacts
brain_dir = Path(r"C:\Users\lam70\.gemini\antigravity\brain\56beb8ea-a685-4296-aba0-d1fe9dcd541c")
if brain_dir.exists():
    out_brain = brain_dir / "training_run_today.png"
    shutil.copy(out_logs, out_brain)
    print(f"Copiado a artifact dir: {out_brain}")

plt.close()
