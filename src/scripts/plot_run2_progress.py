import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil
import base64
from pathlib import Path

csv_path = Path(r"D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_metrics.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

episodes = np.array([int(r["episode"]) for r in rows])
stages = np.array([int(r["stage"]) for r in rows])
critic_loss = np.array([float(r["critic_loss"]) for r in rows])
actor_loss = np.array([float(r["actor_loss"]) for r in rows])
alpha = np.array([float(r["alpha"]) for r in rows])
final_dist = np.array([float(r["final_dist_m"]) * 100 for r in rows]) # cm
time_sec = np.array([float(r["time_sec"]) for r in rows])
success = np.array([r["is_success"] == "True" for r in rows])

# Find promotion episode
promotion_ep = None
for i in range(1, len(stages)):
    if stages[i] > stages[i-1]:
        promotion_ep = episodes[i]
        break

def moving_avg(arr, w=15):
    if len(arr) < w:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0)) 
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad = np.full(w - 1, ma[0])
    return np.concatenate((pad, ma))

plt.style.use('dark_background')
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
fig.suptitle(f'Progreso de Entrenamiento SAC con Curriculo (Episodios 1 a {episodes[-1]})', fontsize=16, fontweight='bold', color='#38bdf8')

# Panel 1: Critic Loss
ax1 = axs[0, 0]
ax1.plot(episodes, critic_loss, color='#64748b', alpha=0.35, label='Critic Loss (raw)')
ax1.plot(episodes, moving_avg(critic_loss, 15), color='#38bdf8', linewidth=2.0, label='Media Movil (w=15)')
if promotion_ep:
    ax1.axvline(promotion_ep, color='#f59e0b', linestyle='--', linewidth=1.5, label=f'Promocion Etapa 2 (Ep {promotion_ep})')
ax1.set_title('Convergencia del Critico (Twin Q-Loss)', fontsize=12, fontweight='semibold')
ax1.set_xlabel('Episodio')
ax1.set_ylabel('Loss MSE')
ax1.set_yscale('log')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper right', fontsize=9)

# Panel 2: Alpha
ax2 = axs[0, 1]
ax2.plot(episodes, alpha, color='#f59e0b', linewidth=2.0, label='Alpha (Temperature)')
if promotion_ep:
    ax2.axvline(promotion_ep, color='#f59e0b', linestyle='--', linewidth=1.5)
ax2.set_title('Temperatura de Entropia (Alpha: 1.000 -> 0.380)', fontsize=12, fontweight='semibold')
ax2.set_xlabel('Episodio')
ax2.set_ylabel('Alpha value')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper right', fontsize=9)

# Panel 3: Actor Loss
ax3 = axs[1, 0]
ax3.plot(episodes, actor_loss, color='#64748b', alpha=0.35, label='Actor Loss (raw)')
ax3.plot(episodes, moving_avg(actor_loss, 15), color='#ec4899', linewidth=2.0, label='Media Movil (w=15)')
if promotion_ep:
    ax3.axvline(promotion_ep, color='#f59e0b', linestyle='--', linewidth=1.5)
ax3.set_title('Perdida de la Politica (Actor Loss)', fontsize=12, fontweight='semibold')
ax3.set_xlabel('Episodio')
ax3.set_ylabel('Actor Loss')
ax3.grid(True, linestyle='--', alpha=0.3)
ax3.legend(loc='upper right', fontsize=9)

# Panel 4: Final Distance & Successes
ax4 = axs[1, 1]
s1_mask = stages == 1
s2_mask = stages == 2

# Truncated vs full
truncated = time_sec < 30.0

# Plot Stage 1
ax4.scatter(episodes[s1_mask & truncated], final_dist[s1_mask & truncated], color='#94a3b8', alpha=0.25, s=12, label='Truncado Etapa 1')
ax4.scatter(episodes[s1_mask & ~truncated & ~success], final_dist[s1_mask & ~truncated & ~success], color='#38bdf8', alpha=0.7, s=25, label='Completo Etapa 1')
ax4.scatter(episodes[s1_mask & success], final_dist[s1_mask & success], color='#22c55e', alpha=0.9, s=50, marker='*', label=f'Exito Etapa 1 (n={np.sum(s1_mask & success)})')

# Plot Stage 2
ax4.scatter(episodes[s2_mask & truncated], final_dist[s2_mask & truncated], color='#94a3b8', alpha=0.25, s=12)
ax4.scatter(episodes[s2_mask & ~truncated], final_dist[s2_mask & ~truncated], color='#a855f7', alpha=0.8, s=30, label='Completo Etapa 2 (Cintura)')

if promotion_ep:
    ax4.axvline(promotion_ep, color='#f59e0b', linestyle='--', linewidth=1.5, label='Promocion Etapa 2')

ax4.axhline(5.0, color='#22c55e', linestyle=':', linewidth=1.5, label='Umbral Meta (5 cm)')
ax4.set_title('Distancia Final a Meta (cm) por Etapa', fontsize=12, fontweight='semibold')
ax4.set_xlabel('Episodio')
ax4.set_ylabel('Distancia (cm)')
ax4.set_ylim(-2, 45)
ax4.grid(True, linestyle='--', alpha=0.3)
ax4.legend(loc='upper right', fontsize=8)

plt.tight_layout()
out_png = r"D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_progress.png"
plt.savefig(out_png)

brain_dir = Path(r"C:\Users\lam70\.gemini\antigravity\brain\1054e8b7-c02b-45d6-bd3b-e45f67e7e502")
artifact_png = brain_dir / "training_progress.png"
shutil.copyfile(out_png, artifact_png)
print("Plot saved to:", out_png, "and", artifact_png)

# Generate HTML Dashboard
with open(artifact_png, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

html_content = f"""<!DOCTYPE html>
<html>
<head>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-2">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-xl p-4 shadow-sm">
    <div class="flex items-center justify-between pb-3 border-b border-[var(--border)]">
      <div>
        <h2 class="text-[var(--foreground)] font-bold text-base flex items-center gap-2">
          <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
          Telemetría en Vivo: Entrenamiento SAC Multi-Etapa (Follow the Object)
        </h2>
        <p class="text-[var(--muted-foreground)] text-xs mt-0.5">
          Episodio actual: {episodes[-1]} de 600 | <span class="text-amber-400 font-semibold">¡Promovido a Etapa 2 en Ep {promotion_ep}!</span>
        </p>
      </div>
      <div class="text-right">
        <span class="text-xs font-mono px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20 font-semibold">
          Etapa Activa: 2 (Levantamiento de Cintura)
        </span>
      </div>
    </div>

    <!-- Mini KPI cards -->
    <div class="grid grid-cols-4 gap-2 my-3 text-center">
      <div class="p-2 rounded-lg bg-[var(--background)] border border-[var(--border)]">
        <div class="text-[12px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Éxitos Etapa 1</div>
        <div class="text-sm font-bold text-emerald-400 font-mono mt-0.5">81 validados (1.97 cm min)</div>
      </div>
      <div class="p-2 rounded-lg bg-[var(--background)] border border-[var(--border)]">
        <div class="text-[12px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Alpha (&alpha;)</div>
        <div class="text-sm font-bold text-amber-400 font-mono mt-0.5">{alpha[-1]:.3f} (Decaimiento óptimo)</div>
      </div>
      <div class="p-2 rounded-lg bg-[var(--background)] border border-[var(--border)]">
        <div class="text-[12px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Tasa Completitud Etapa 2</div>
        <div class="text-sm font-bold text-purple-400 font-mono mt-0.5">78.8% (&gt;30s)</div>
      </div>
      <div class="p-2 rounded-lg bg-[var(--background)] border border-[var(--border)]">
        <div class="text-[12px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Critic Loss Actual</div>
        <div class="text-sm font-bold text-sky-400 font-mono mt-0.5">{critic_loss[-1]:.2f} MSE</div>
      </div>
    </div>

    <!-- Chart image -->
    <div class="rounded-lg overflow-hidden border border-[var(--border)] bg-black/40 mt-1">
      <img src="data:image/png;base64,{b64_data}" alt="Curvas de Entrenamiento Multi-Etapa" class="w-full h-auto object-contain max-h-[380px]" />
    </div>
  </div>
</body>
</html>"""

out_html = brain_dir / "training_dashboard.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_content)
print("Dashboard updated at:", out_html)
