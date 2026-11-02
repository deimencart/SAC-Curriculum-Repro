import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil

log_path = Path("D:/genesis_ws/phase_5_gym_env/cloth_curriculum_rl/logs/training_metrics.jsonl")
brain_dir = Path(r"C:\Users\lam70\.gemini\antigravity\brain\2dae13c3-0596-4949-8bff-f09e50776129")

with open(log_path, "r") as f:
    lines = [json.loads(line) for line in f]

# Extract last 500 episodes (the full run)
run_data = lines[-500:]

episodes = np.array([x["episode"] for x in run_data])
rewards = np.array([x["ep_reward"] for x in run_data])
critic_loss = np.array([x["critic_loss"] for x in run_data])
actor_loss = np.array([x["actor_loss"] for x in run_data])
alpha = np.array([x["alpha"] for x in run_data])
waist_err = np.array([x["final_dist_m"] * 100 for x in run_data]) # cm
span_xy = np.array([x["fold_dist_xy"] * 100 for x in run_data]) # cm
contact = np.array([x["contact_ratio"] * 100 for x in run_data]) # %
ankle_err = np.array([x.get("alignment_error_m", 0.0) * 100 for x in run_data]) # cm

def moving_avg(arr, w=20):
    if len(arr) < w:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)
    pad = np.full(w - 1, ma[0])
    return np.concatenate((pad, ma))

r_ma = moving_avg(rewards, 25)
c_loss_ma = moving_avg(critic_loss, 25)
a_loss_ma = moving_avg(actor_loss, 25)
waist_ma = moving_avg(waist_err, 25)
span_ma = moving_avg(span_xy, 25)
contact_ma = moving_avg(contact, 25)
ankle_ma = moving_avg(ankle_err, 25)

plt.style.use("dark_background")
fig, axs = plt.subplots(4, 2, figsize=(16, 18), dpi=150)
fig.suptitle(
    "Etapa 2: Dinámica de Entrenamiento SAC (500 Episodios Completos)\n"
    "Estiramiento Longitudinal y Posado Plano de Cintura (Currículo Inverso - Nivel 1)",
    fontsize=16,
    fontweight="bold",
    color="#38bdf8",
    y=0.995,
)

# 1. Recompensa
ax = axs[0, 0]
ax.plot(episodes, rewards, color="#38bdf8", alpha=0.25, label="Recompensa Episodio")
ax.plot(episodes, r_ma, color="#0284c7", linewidth=2.5, label="Media Móvil (w=25)")
ax.axhline(0.0, color="#ef4444", linestyle="--", alpha=0.7, label="Umbral R = 0")
ax.set_title("Recompensa por Episodio (Retorno Acumulado)", fontweight="bold", color="#f8fafc")
ax.set_xlabel("Episodio")
ax.set_ylabel("Recompensa")
ax.grid(True, alpha=0.2)
ax.legend(loc="lower right")

# 2. Critic Loss
ax = axs[0, 1]
ax.plot(episodes, critic_loss, color="#f97316", alpha=0.25, label="Critic Loss (raw)")
ax.plot(episodes, c_loss_ma, color="#ea580c", linewidth=2.5, label="Media Móvil (w=25)")
ax.set_title("Pérdida del Crítico (Critic Loss - MSE Q-Bellman)", fontweight="bold", color="#f8fafc")
ax.set_xlabel("Episodio")
ax.set_ylabel("Loss")
ax.grid(True, alpha=0.2)
ax.legend(loc="upper right")

# 3. Actor Loss
ax = axs[1, 0]
ax.plot(episodes, actor_loss, color="#a855f7", alpha=0.25, label="Actor Loss (raw)")
ax.plot(episodes, a_loss_ma, color="#9333ea", linewidth=2.5, label="Media Móvil (w=25)")
ax.set_title("Pérdida del Actor (Policy Gradient con Entropía)", fontweight="bold", color="#f8fafc")
ax.set_xlabel("Episodio")
ax.set_ylabel("Loss")
ax.grid(True, alpha=0.2)
ax.legend(loc="upper right")

# 4. Temperatura Alpha
ax = axs[1, 1]
ax.plot(episodes, alpha, color="#10b981", linewidth=2.5, label=r"$\alpha$ (Coeficiente Entropía SAC)")
ax.set_title(r"Enfriamiento de Temperatura SAC ($\alpha$ de Auto-Tuning)", fontweight="bold", color="#f8fafc")
ax.set_xlabel("Episodio")
ax.set_ylabel(r"$\alpha$")
ax.grid(True, alpha=0.2)
ax.legend(loc="upper right")

# 5. Error Posicionamiento Cintura
ax = axs[2, 0]
ax.plot(episodes, waist_err, color="#fbbf24", alpha=0.25, label="Error Cintura (raw)")
ax.plot(episodes, waist_ma, color="#d97706", linewidth=2.5, label="Media Móvil (w=25)")
ax.axhline(4.5, color="#ef4444", linestyle="--", linewidth=1.5, label="Tolerancia Máx (4.5 cm)")
ax.set_title("Precisión de Colocación de la Cintura (Error al Target, cm)", fontweight="bold", color="#f8fafc")
ax.set_xlabel("Episodio")
ax.set_ylabel("Error (cm)")
ax.grid(True, alpha=0.2)
ax.legend(loc="upper right")

# 6. Envergadura / Extensión Longitudinal
ax = axs[2, 1]
ax.plot(episodes, span_xy, color="#06b6d4", alpha=0.25, label="Envergadura Perneras (raw)")
ax.plot(episodes, span_ma, color="#0891b2", linewidth=2.5, label="Media Móvil (w=25)")
ax.axhline(57.1, color="#22c55e", linestyle="--", linewidth=1.8, label="Longitud Nominal Extendida (57.1 cm)")
ax.axhspan(50.6, 63.6, color="#22c55e", alpha=0.10, label="Banda Éxito Span (50.6 - 63.6 cm)")
ax.set_title("Envergadura Longitudinal del Pantalón ($L_{span}$, cm)", fontweight="bold", color="#f8fafc")
ax.set_xlabel("Episodio")
ax.set_ylabel("Longitud (cm)")
ax.grid(True, alpha=0.2)
ax.legend(loc="lower right")

# 7. Ratio de Contacto Plano
ax = axs[3, 0]
ax.plot(episodes, contact, color="#ec4899", alpha=0.25, label="Contacto Mesa (raw)")
ax.plot(episodes, contact_ma, color="#db2777", linewidth=2.5, label="Media Móvil (w=25)")
ax.axhline(40.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Umbral Mínimo Éxito (40%)")
ax.set_title("Ratio de Contacto con la Mesa ($C_{ratio}$, %)", fontweight="bold", color="#f8fafc")
ax.set_xlabel("Episodio")
ax.set_ylabel("Contacto (%)")
ax.grid(True, alpha=0.2)
ax.legend(loc="lower right")

# 8. Separación de Tobillos (Diagnóstico Curricular)
ax = axs[3, 1]
ax.plot(episodes, ankle_err, color="#e11d48", alpha=0.25, label="Distancia Tobillos (raw)")
ax.plot(episodes, ankle_ma, color="#be123c", linewidth=2.5, label="Media Móvil (w=25)")
ax.axhline(4.0, color="#f59e0b", linestyle="--", linewidth=2.0, label="Umbral Estricto Promoción (4.0 cm)")
ax.axhline(4.6, color="#3b82f6", linestyle=":", linewidth=1.8, label="Separación Inicial Pose C (4.6 cm)")
ax.set_title("Separación de Bajos de Pernera ($d_{ankles}$, cm) - Causa de Retención en Nivel 1", fontweight="bold", color="#f8fafc")
ax.set_xlabel("Episodio")
ax.set_ylabel("Distancia (cm)")
ax.grid(True, alpha=0.2)
ax.legend(loc="upper right")

plt.tight_layout(rect=[0, 0.01, 1, 0.98])

output_png = Path("D:/genesis_ws/phase_5_gym_env/cloth_curriculum_rl/logs/stage2_500ep_complete_training_analysis.png")
plt.savefig(str(output_png), dpi=150)
plt.close()
print(f"[*] Gráfica guardada en: {output_png}")

# Copy to brain dir
brain_png = brain_dir / "stage2_500ep_complete_training_analysis.png"
shutil.copy(str(output_png), str(brain_png))
print(f"[*] Gráfica copiada al Brain en: {brain_png}")
