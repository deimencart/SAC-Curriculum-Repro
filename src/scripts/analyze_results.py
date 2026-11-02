import csv
import torch

csv_path = r"D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_metrics.csv"
with open(csv_path, "r") as f:
    rows = list(csv.DictReader(f))

start_idx = 0
for i, r in enumerate(rows):
    if r["episode"] == "1":
        start_idx = i
rows = rows[start_idx:]

successes = [r for r in rows if r["is_success"] == "True"]
completed_runs = [r for r in rows if float(r["time_sec"]) > 30.0]
low_dist_runs = [r for r in rows if float(r["final_dist_m"]) < 0.10]

print(f"Total episodes in run: {len(rows)}")
print(f"Total successful episodes (dist < 5cm & flagged True): {len(successes)}")
print(f"Total episodes with dist < 10cm: {len(low_dist_runs)}")
print(f"Total completed runs (> 30s): {len(completed_runs)}")

print("\nDetailed Successful Episodes:")
for s in successes:
    print(f"Ep: {s['episode']}, Dist: {float(s['final_dist_m'])*100:.2f} cm, Reward: {float(s['ep_reward']):.2f}, Time: {float(s['time_sec']):.2f}s")

print("\nDetailed Low Dist (< 10cm) Episodes:")
for s in low_dist_runs:
    print(f"Ep: {s['episode']}, Dist: {float(s['final_dist_m'])*100:.2f} cm, Time: {float(s['time_sec']):.2f}s, Success: {s['is_success']}")

# Inspect checkpoints
print("\n--- Checkpoint Information ---")
for ckpt_name in ["sac_best_stage1.pt", "sac_latest.pt"]:
    ckpt_path = f"D:/genesis_ws/phase_5_gym_env/cloth_curriculum_rl/checkpoints/{ckpt_name}"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        print(f"File: {ckpt_name}")
        for k in ckpt.keys():
            if k not in ["actor_state_dict", "critic1_state_dict", "critic2_state_dict"]:
                print(f"  {k}: {ckpt[k]}")
    except Exception as e:
        print(f"Error loading {ckpt_name}: {e}")
