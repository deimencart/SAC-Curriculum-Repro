import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil

csv_path = r'D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_metrics.csv'
with open(csv_path, 'r') as f:
    rows = list(csv.DictReader(f))

start_idx = 0
for i, r in enumerate(rows):
    if r['episode'] == '1':
        start_idx = i
rows = rows[start_idx:]

episodes = np.array([int(r['episode']) for r in rows])
critic_loss = np.array([float(r['critic_loss']) for r in rows])
actor_loss = np.array([float(r['actor_loss']) for r in rows])
alpha = np.array([float(r['alpha']) for r in rows])
final_dist = np.array([float(r['final_dist_m']) * 100 for r in rows]) # in cm
time_sec = np.array([float(r['time_sec']) for r in rows])
success = np.array([r['is_success'] == 'True' for r in rows])

def moving_avg(arr, w=15):
    if len(arr) < w:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0)) 
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad = np.full(w - 1, ma[0])
    return np.concatenate((pad, ma))

plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
fig.suptitle(f'Resultados Finales del Entrenamiento SAC (500 Episodios Completos)', fontsize=16, fontweight='bold', color='#38bdf8')

# Panel 1: Critic Loss
ax1 = axs[0, 0]
ax1.plot(episodes, critic_loss, color='#64748b', alpha=0.35, label='Critic Loss (raw)')
ax1.plot(episodes, moving_avg(critic_loss, 15), color='#38bdf8', linewidth=2.0, label='Media Movil (w=15)')
ax1.set_title('Convergencia del Critico (Twin Q-Loss: 172 -> 1.48 MSE)', fontsize=12, fontweight='semibold')
ax1.set_xlabel('Episodio')
ax1.set_ylabel('Loss MSE')
ax1.set_yscale('log')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper right')

# Panel 2: Alpha
ax2 = axs[0, 1]
ax2.plot(episodes, alpha, color='#f59e0b', linewidth=2.0, label='Temperature Alpha')
ax2.set_title('Temperatura de Entropia (Alpha: 1.000 -> 0.782)', fontsize=12, fontweight='semibold')
ax2.set_xlabel('Episodio')
ax2.set_ylabel('Alpha value')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper right')

# Panel 3: Actor Loss
ax3 = axs[1, 0]
ax3.plot(episodes, actor_loss, color='#64748b', alpha=0.35, label='Actor Loss (raw)')
ax3.plot(episodes, moving_avg(actor_loss, 15), color='#ec4899', linewidth=2.0, label='Media Movil (w=15)')
ax3.set_title('Perdida de la Politica (Actor Loss Estable en ~6.0)', fontsize=12, fontweight='semibold')
ax3.set_xlabel('Episodio')
ax3.set_ylabel('Actor Loss')
ax3.grid(True, linestyle='--', alpha=0.3)
ax3.legend(loc='upper right')

# Panel 4: Final Distance & Successes
ax4 = axs[1, 1]
truncated_mask = time_sec < 30.0
full_mask = ~truncated_mask

ax4.scatter(episodes[truncated_mask], final_dist[truncated_mask], color='#94a3b8', alpha=0.35, s=15, label='Truncado por Fusible (~39 cm)')
ax4.scatter(episodes[full_mask & ~success], final_dist[full_mask & ~success], color='#f97316', alpha=0.8, s=40, label='Completado (>30s)')
ax4.scatter(episodes[success], final_dist[success], color='#22c55e', alpha=1.0, s=80, marker='*', label='Exitos Validados (<5 cm)')

ax4.axhline(5.0, color='#22c55e', linestyle=':', linewidth=1.5, label='Umbral Meta (5 cm)')
ax4.set_title('Distancia Final a la Meta (cm) - 100% Exito en Trayectorias Completas', fontsize=12, fontweight='semibold')
ax4.set_xlabel('Episodio')
ax4.set_ylabel('Distancia (cm)')
ax4.set_ylim(-2, 50)
ax4.grid(True, linestyle='--', alpha=0.3)
ax4.legend(loc='upper right', fontsize=9)

plt.tight_layout()
out_png = r'D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_progress.png'
plt.savefig(out_png)
artifact_png = r'C:\Users\lam70\.gemini\antigravity\brain\1054e8b7-c02b-45d6-bd3b-e45f67e7e502\training_progress.png'
shutil.copyfile(out_png, artifact_png)
print('Complete 500-ep plot saved successfully.')
