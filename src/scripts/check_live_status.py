import csv
from pathlib import Path

csv_path = Path(r"D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_metrics.csv")
with open(csv_path, "r") as f:
    rows = list(csv.DictReader(f))

stages = [int(r["stage"]) for r in rows]
print("Total rows logged:", len(rows))
print("Distinct stages observed:", sorted(list(set(stages))))

stage1_rows = [r for r in rows if r["stage"] == "1"]
stage2_rows = [r for r in rows if r["stage"] == "2"]

print(f"Stage 1 episodes count: {len(stage1_rows)}")
print(f"Stage 2 episodes count: {len(stage2_rows)}")

if stage1_rows:
    s1_successes = [r for r in stage1_rows if r["is_success"] == "True"]
    s1_full = [r for r in stage1_rows if float(r["time_sec"]) > 30.0]
    print(f"\n--- STAGE 1 RESUMEN ---")
    print(f"Total episodios: {len(stage1_rows)}")
    print(f"Episodios con horizonte completo (>30s): {len(s1_full)} ({len(s1_full)/len(stage1_rows)*100:.1f}%)")
    print(f"Éxitos marcados: {len(s1_successes)}")
    if s1_full:
        dists = [float(r["final_dist_m"]) * 100 for r in s1_full]
        print(f"Distancia mínima alcanzada: {min(dists):.2f} cm")
        print(f"Distancia media: {sum(dists)/len(dists):.2f} cm")

if stage2_rows:
    s2_successes = [r for r in stage2_rows if r["is_success"] == "True"]
    s2_full = [r for r in stage2_rows if float(r["time_sec"]) > 30.0]
    print(f"\n--- STAGE 2 RESUMEN ---")
    print(f"Total episodios: {len(stage2_rows)}")
    print(f"Episodios con horizonte completo (>30s): {len(s2_full)} ({len(s2_full)/len(stage2_rows)*100:.1f}%)")
    print(f"Éxitos marcados: {len(s2_successes)}")
    if s2_full:
        dists = [float(r["final_dist_m"]) * 100 for r in s2_full]
        print(f"Distancia mínima alcanzada: {min(dists):.2f} cm")
        print(f"Distancia media: {sum(dists)/len(dists):.2f} cm")

print("\n--- TRANSICIONES DE ETAPA ---")
for i in range(1, len(rows)):
    if rows[i]["stage"] != rows[i-1]["stage"]:
        ep = rows[i]["episode"]
        print(f"Episodio {ep}: Transición de Etapa {rows[i-1]['stage']} -> Etapa {rows[i]['stage']}")

print(f"\nÚltimo episodio registrado: {rows[-1]['episode']} (Etapa {rows[-1]['stage']})")
print(f"Critic Loss actual: {float(rows[-1]['critic_loss']):.4f} | Alpha: {float(rows[-1]['alpha']):.4f}")
