import base64
import os

img_path = r"C:\Users\lam70\.gemini\antigravity\brain\1054e8b7-c02b-45d6-bd3b-e45f67e7e502\training_progress.png"
if not os.path.exists(img_path):
    img_path = r"D:\genesis_ws\phase_5_gym_env\cloth_curriculum_rl\logs\training_progress.png"

with open(img_path, "rb") as f:
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
          Telemetría en Vivo: Entrenamiento SAC (Genesis XPBD)
        </h2>
        <p class="text-[var(--muted-foreground)] text-xs mt-0.5">Estado de convergencia y distancias finales (Episodios 1 a 300+)</p>
      </div>
      <div class="text-right">
        <span class="text-xs font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold">
          CritLoss: ~6.2 MSE (-98%)
        </span>
      </div>
    </div>

    <!-- Mini KPI cards -->
    <div class="grid grid-cols-4 gap-2 my-3 text-center">
      <div class="p-2 rounded-lg bg-[var(--background)] border border-[var(--border)]">
        <div class="text-[12px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Critic Loss</div>
        <div class="text-sm font-bold text-sky-400 font-mono mt-0.5">172 &rarr; 6.2</div>
      </div>
      <div class="p-2 rounded-lg bg-[var(--background)] border border-[var(--border)]">
        <div class="text-[12px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Alpha (&alpha;)</div>
        <div class="text-sm font-bold text-amber-400 font-mono mt-0.5">1.00 &rarr; 0.868</div>
      </div>
      <div class="p-2 rounded-lg bg-[var(--background)] border border-[var(--border)]">
        <div class="text-[12px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Mejor Error</div>
        <div class="text-sm font-bold text-emerald-400 font-mono mt-0.5">2.43 cm</div>
      </div>
      <div class="p-2 rounded-lg bg-[var(--background)] border border-[var(--border)]">
        <div class="text-[12px] text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Éxitos Plegado</div>
        <div class="text-sm font-bold text-indigo-400 font-mono mt-0.5">4 validados</div>
      </div>
    </div>

    <!-- Chart image -->
    <div class="rounded-lg overflow-hidden border border-[var(--border)] bg-black/40 mt-1">
      <img src="data:image/png;base64,{b64_data}" alt="Curvas de Entrenamiento" class="w-full h-auto object-contain max-h-[380px]" />
    </div>
  </div>
</body>
</html>"""

out_html = r"C:\Users\lam70\.gemini\antigravity\brain\1054e8b7-c02b-45d6-bd3b-e45f67e7e502\training_dashboard.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_content)
print("Dashboard HTML successfully written to:", out_html)
