"""
generate_sac_drawio.py
======================
Generates a comprehensive, production-grade Draw.io (.drawio) XML architecture diagram
for the Soft Actor-Critic (SAC) residual cloth folding system implemented in Phase 5.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

def build_drawio_xml():
    mxfile = ET.Element('mxfile', host='app.diagrams.net', agent='Antigravity Agent', version='24.0.0')
    diagram = ET.SubElement(mxfile, 'diagram', id='sac_architecture_phase5', name='SAC Residual Architecture Phase 5')
    mxGraphModel = ET.SubElement(
        diagram, 'mxGraphModel',
        dx='2000', dy='1400', grid='1', gridSize='10', guides='1',
        tooltips='1', connect='1', arrows='1', fold='1', page='1',
        pageScale='1', pageWidth='1800', pageHeight='1400', math='1', shadow='0'
    )
    root = ET.SubElement(mxGraphModel, 'root')

    ET.SubElement(root, 'mxCell', id='0')
    ET.SubElement(root, 'mxCell', id='1', parent='0')

    cells = []

    def add_node(cid, val, style, x, y, w, h, parent='1'):
        c = ET.SubElement(root, 'mxCell', id=str(cid), value=val, style=style, vertex='1', parent=str(parent))
        ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as': 'geometry'})
        return c

    def add_edge(eid, val, style, src, tgt, waypoints=None, parent='1'):
        e = ET.SubElement(root, 'mxCell', id=str(eid), value=val, style=style, edge='1', parent=str(parent), source=str(src), target=str(tgt))
        geo = ET.SubElement(e, 'mxGeometry', relative='1', **{'as': 'geometry'})
        if waypoints:
            pts = ET.SubElement(geo, 'Array', **{'as': 'points'})
            for wx, wy in waypoints:
                ET.SubElement(pts, 'mxPoint', x=str(wx), y=str(wy))
        return e

    # =========================================================================
    # STYLES DEFINITION
    # =========================================================================
    # Banner
    style_banner = (
        "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#0F172A;strokeColor=#1E293B;"
        "fontColor=#F8FAFC;fontStyle=1;fontSize=18;align=center;verticalAlign=middle;"
        "shadow=1;strokeWidth=2;"
    )

    # Containers (Subsystems)
    def container_style(fill, stroke, title_color):
        return (
            f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;verticalAlign=top;align=center;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=2;fontStyle=1;fontSize=14;"
            f"fontColor={title_color};spacingTop=12;dashed=0;shadow=0;"
        )

    # Standard Node Inside Subsystem
    def node_style(fill, stroke, font_color="#1E293B", bold=False, arc=8):
        b = "1" if bold else "0"
        return (
            f"rounded=1;arcSize={arc};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"fontColor={font_color};fontStyle={b};fontSize=11;align=center;verticalAlign=middle;"
            f"strokeWidth=1.5;spacing=4;shadow=0;"
        )

    # Edge Style
    def edge_style(stroke="#475569", width=2, dashed=0, startArrow="none", endArrow="classic"):
        d = "1" if dashed else "0"
        return (
            f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;"
            f"strokeColor={stroke};strokeWidth={width};dashed={d};startArrow={startArrow};"
            f"endArrow={endArrow};fontSize=10;fontColor=#334155;labelBackgroundColor=#FFFFFF;"
        )

    # =========================================================================
    # 0. HEADER BANNER
    # =========================================================================
    banner_text = (
        "<b>ARQUITECTURA SAC RESIDUAL & CURRICULUM RL — FASE 5</b><br>"
        "<span style='font-size:12px;font-weight:normal;color:#94A3B8;'>"
        "Doblado Dinámico de Ropa en Génesis World (XPBD) | Soft Actor-Critic Continuo (22D Obs → 3D Residual Act) | Gestor de Fricción Adaptativo"
        "</span>"
    )
    add_node(100, banner_text, style_banner, 40, 20, 1680, 60)

    # =========================================================================
    # 1. SUBSYSTEM: ENTORNO Y SIMULACIÓN FÍSICA (GENESIS XPBD BACKEND)
    # Bounds: x=40, y=100, w=480, h=920
    # =========================================================================
    add_node(
        1000,
        "<b>SUBSISTEMA 1: ENTORNO GYMNASIUM & FÍSICA GENESIS XPBD</b><br>"
        "<span style='font-size:10px;font-weight:normal;color:#0F766E;'>DynamicFoldingEnv & ClothGenesisBackend</span>",
        container_style("#F0FDFA", "#0D9488", "#0F766E"),
        40, 100, 480, 920
    )

    # 1.1 Genesis Scene & Physics Engine
    node_genesis = (
        "<b>Simulador Físico: Genesis World (v1.4.0 / 0.3.13)</b><br>"
        "• Backend: GPU (Taichi JIT / CUDA)<br>"
        "• Solver de Partículas: XPBD (<tt>gs.materials.PBD.Cloth</tt>)<br>"
        "• Timestep: &Delta;t = 0.004 s (250 Hz) | Substeps: 6 &ndash; 12<br>"
        "• Iteraciones: Stretch=10, Bending=10 | Tamaño partícula = 8 mm<br>"
        "• Entornos Paralelos: N<sub>envs</sub> = 4 (Grilla 2x2, >120 FPS combinados)"
    )
    add_node(1101, node_genesis, node_style("#CCFBF1", "#0D9488", bold=False), 60, 160, 440, 95)

    # 1.2 Cloth Mesh & Keypoints
    node_mesh = (
        "<b>Prenda Deformable & Puntos Semánticos</b><br>"
        "• Malla OBJ: Pantalón FoldNet (430) / ClothesNet (PS_004)<br>"
        "• Topología: ~4.500 partículas deformables XPBD<br>"
        "• 7 Keypoints Anatómicos Canónicos:<br>"
        "  <tt>[l_corner, r_corner, crotch, top_ctr_f, top_ctr_b, l_leg_o, r_leg_o]</tt><br>"
        "• Coordenadas Métricas 3D en Sistema de Coordenadas Local del Entorno"
    )
    add_node(1102, node_mesh, node_style("#FFFFFF", "#14B8A6"), 60, 280, 440, 85)

    # 1.3 State Extraction & Vector Builder (Observation Space)
    node_obs_builder = (
        "<b>Extractor del Vector de Observación (s<sub>t</sub> &isin; &real;<sup>22</sup>)</b><br>"
        "• 7 Puntos Clave &times; Posición 3D (x, y, z) = <b>21 dimensiones</b><br>"
        "• Progreso Temporal Normalizado: <b>&tau; = t / T &isin; [0.0, 1.0]</b> (1 dimensión)<br>"
        "• Dimensión Total: <b>s<sub>t</sub> = [p<sub>1</sub>, ..., p<sub>7</sub>, &tau;] &isin; &real;<sup>22</sup></b> (o [B, 22] en multi-env)<br>"
        "• Validación: Cero NaNs, coordenadas locales invariantes al offset de grilla"
    )
    add_node(1103, node_obs_builder, node_style("#FEF3C7", "#D97706", bold=True), 60, 395, 440, 85)

    # 1.4 Nominal Bézier Reference Generator
    node_bezier = (
        "<b>Generador de Trayectoria Nominal de Bézier (C<sup>2</sup> Quintic)</b><br>"
        "• Ley Temporal de Mínimo Tirón: s(&tau;) = 10&tau;<sup>3</sup> - 15&tau;<sup>4</sup> + 6&tau;<sup>5</sup><br>"
        "• Elevación Parabólica: &Aacute;pice h = h<sub>ratio</sub> &middot; ||P<sub>target</sub> - P<sub>0</sub>|| (h<sub>ratio</sub>=0.65)<br>"
        "• Posición Analítica Suave: <b>x<sub>nominal</sub>(&tau;) &isin; &real;<sup>3</sup></b><br>"
        "• Propósito: Guiar el doblado macro libre de inestabilidades cinemáticas"
    )
    add_node(1104, node_bezier, node_style("#EDE9FE", "#8B5CF6"), 60, 510, 440, 85)

    # 1.5 Residual Action Blender
    node_blender = (
        "<b>Mezclador de Acción Residual (Fusión Cinemática)</b><br>"
        "• Acción de la Política SAC: <b>a<sub>t</sub> &isin; [-0.03, +0.03] m</b> (&plusmn;3 cm)<br>"
        "• Comando Cinemático Final: <b>x<sub>cmd</sub> = x<sub>nominal</sub>(&tau;) + a<sub>t</sub></b><br>"
        "• Clamping de Seguridad: np.clip(a<sub>t</sub>, -0.03, 0.03)<br>"
        "• Mecanismo de Control: Desplazamiento rígido acoplado a parche PBD"
    )
    add_node(1105, node_blender, node_style("#FEE2E2", "#EF4444", bold=True), 60, 625, 440, 80)

    # 1.6 Kinematic Patch Step in XPBD
    node_patch_step = (
        "<b>Paso de Integración Cinemática (ClothGenesisBackend)</b><br>"
        "• Parche activo anclado: partículas fijas r<sub>grasp</sub> = 20 mm<br>"
        "• Actualización continua C<sup>1</sup>: set_particles_pos(x<sub>cmd</sub> + offset)<br>"
        "• Integración XPBD: scene.step() &times; substeps (sin violar tensiones)"
    )
    add_node(1106, node_patch_step, node_style("#FFFFFF", "#0D9488"), 60, 735, 440, 65)

    # 1.7 Reward Engine & Termination Metrics
    node_reward = (
        "<b>Función de Recompensa Cóncava (Li et al., 2025)</b><br>"
        "• Distancia Euclídea: d = ||p<sub>activo</sub> - p<sub>target</sub>||<br>"
        "• Normalización: d<sub>norm</sub> = d / (&delta;<sub>threshold</sub> + 10<sup>-7</sup>), con &delta; = 0.05 m<br>"
        "• <b>R(s, a) = 1.0 - (d<sub>norm</sub>)<sup>0.4</sup></b> &nbsp;si d &le; &delta;, sino <b>-1.0</b><br>"
        "• Término: Truncado a horizonte H=20 pasos | Release & Settle: 40 steps"
    )
    add_node(1107, node_reward, node_style("#ECFDF5", "#059669", bold=True), 60, 830, 440, 80)

    # Internal Edges in Subsystem 1
    add_edge(1201, "", edge_style("#0D9488"), 1101, 1102)
    add_edge(1202, "Lectura de vértices", edge_style("#0D9488"), 1102, 1103)
    add_edge(1203, "Progreso &tau;", edge_style("#8B5CF6"), 1103, 1104)
    add_edge(1204, "x_nominal(&tau;)", edge_style("#8B5CF6"), 1104, 1105)
    add_edge(1205, "x_cmd", edge_style("#EF4444"), 1105, 1106)
    add_edge(1206, "PBD Step", edge_style("#0D9488"), 1106, 1107)

    # =========================================================================
    # 2. SUBSYSTEM: RED ACTOR GAUSSIANA ESTOCÁSTICA (PI_PHI)
    # Bounds: x=550, y=100, w=520, h=440
    # =========================================================================
    add_node(
        2000,
        "<b>SUBSISTEMA 2: ACTOR GAUSSIANO CON COMPRESIÓN TANH (&pi;<sub>&phi;</sub>)</b><br>"
        "<span style='font-size:10px;font-weight:normal;color:#6B21A8;'>GaussianActor &middot; sac_policy.py</span>",
        container_style("#FAF5FF", "#9333EA", "#6B21A8"),
        550, 100, 520, 440
    )

    # 2.1 State Input
    node_actor_in = (
        "<b>Entrada de Estado</b>: s<sub>t</sub> &isin; &real;<sup>22</sup> (7 keypoints &times; 3D + &tau;)"
    )
    add_node(2101, node_actor_in, node_style("#F3E8FF", "#A855F7", bold=True), 570, 155, 480, 35)

    # 2.2 Shared MLP Trunk
    node_actor_trunk = (
        "<b>Tronco Compartido (MLP de 2 Capas Ocultas)</b><br>"
        "• Linear(in_features=22, out_features=256) &rarr; ReLU()<br>"
        "• Linear(in_features=256, out_features=256) &rarr; ReLU()<br>"
        "• Representación Latente: h<sub>actor</sub> &isin; &real;<sup>256</sup>"
    )
    add_node(2102, node_actor_trunk, node_style("#FFFFFF", "#9333EA"), 570, 205, 480, 60)

    # 2.3 Dual Heads (Mean & Log-Std)
    node_head_mu = (
        "<b>Cabeza Media (&mu;)</b><br>"
        "• Linear(256, 3)<br>"
        "• Vector &mu; &isin; &real;<sup>3</sup>"
    )
    add_node(2103, node_head_mu, node_style("#EDE9FE", "#7C3AED"), 570, 280, 230, 50)

    node_head_std = (
        "<b>Cabeza Log-Std (log &sigma;)</b><br>"
        "• Linear(256, 3) &rarr; clamp([-20, 2])<br>"
        "• &sigma; = exp(log &sigma;) &isin; &real;<sup>3</sup>"
    )
    add_node(2104, node_head_std, node_style("#EDE9FE", "#7C3AED"), 820, 280, 230, 50)

    # 2.4 Reparameterization Trick
    node_reparam = (
        "<b>Truco de Reparametrización Gaussiana</b><br>"
        "u = &mu; + &sigma; &odot; &epsilon;, &nbsp;&nbsp;&epsilon; ~ &Nu;(0, I) &isin; &real;<sup>3</sup><br>"
        "<span style='font-size:10px;color:#6B21A8;'>Permite retropropagar gradientes &nabla;<sub>&phi;</sub> a través del muestreo</span>"
    )
    add_node(2105, node_reparam, node_style("#F5F3FF", "#6D28D9", bold=True), 570, 345, 480, 50)

    # 2.5 Squashing, Scaling & Log-Prob Correction
    node_squash = (
        "<b>Compresión Tanh & Corrección de Densidad (Jacobiano)</b><br>"
        "• Acción no escalada: &atilde; = tanh(u) &isin; (-1, 1)<sup>3</sup><br>"
        "• <b>Acción Residual Escalada: a<sub>t</sub> = &atilde; &middot; 0.03 m &isin; [-0.03, +0.03] m</b><br>"
        "• Corrección: log &pi;(a|s) = log &Nu;(u|&mu;,&sigma;) - &sum;<sub>i=1</sub><sup>3</sup> log(1 - &atilde;<sub>i</sub><sup>2</sup> + 10<sup>-7</sup>)"
    )
    add_node(2106, node_squash, node_style("#FEF3C7", "#D97706", bold=True), 570, 410, 480, 65)

    # Internal Edges Actor
    add_edge(2201, "", edge_style("#9333EA"), 2101, 2102)
    add_edge(2202, "", edge_style("#9333EA"), 2102, 2103)
    add_edge(2203, "", edge_style("#9333EA"), 2102, 2104)
    add_edge(2204, "&mu;", edge_style("#7C3AED"), 2103, 2105)
    add_edge(2205, "&sigma;", edge_style("#7C3AED"), 2104, 2105)
    add_edge(2206, "u", edge_style("#6D28D9"), 2105, 2106)

    # =========================================================================
    # 3. SUBSYSTEM: REDES TWIN CRITIC (Q1, Q2) & REDES TARGET
    # Bounds: x=550, y=560, w=520, h=460
    # =========================================================================
    add_node(
        3000,
        "<b>SUBSISTEMA 3: CRÍTICOS MELLIZOS & DIANA (TWIN-Q & TARGET CRITICS)</b><br>"
        "<span style='font-size:10px;font-weight:normal;color:#1E40AF;'>TwinCritic &middot; Clipped Double Q-Learning</span>",
        container_style("#EFF6FF", "#2563EB", "#1E40AF"),
        550, 560, 520, 460
    )

    # 3.1 State-Action Concat
    node_critic_in = (
        "<b>Concatenación Estado-Acción</b>: [s<sub>t</sub>, a<sub>t</sub>] &isin; &real;<sup>22 + 3</sup> = &real;<sup>25</sup>"
    )
    add_node(3101, node_critic_in, node_style("#DBEAFE", "#3B82F6", bold=True), 570, 615, 480, 35)

    # 3.2 Twin Critic Networks
    node_q1 = (
        "<b>Red Crítico Q<sub>&theta;1</sub>(s, a)</b><br>"
        "• Linear(25, 256) &rarr; ReLU()<br>"
        "• Linear(256, 256) &rarr; ReLU()<br>"
        "• Linear(256, 1) &rarr; Escalar Q<sub>1</sub>"
    )
    add_node(3102, node_q1, node_style("#FFFFFF", "#2563EB"), 570, 665, 230, 75)

    node_q2 = (
        "<b>Red Crítico Q<sub>&theta;2</sub>(s, a)</b><br>"
        "• Linear(25, 256) &rarr; ReLU()<br>"
        "• Linear(256, 256) &rarr; ReLU()<br>"
        "• Linear(256, 1) &rarr; Escalar Q<sub>2</sub>"
    )
    add_node(3103, node_q2, node_style("#FFFFFF", "#2563EB"), 820, 665, 230, 75)

    # 3.3 Target Networks
    node_target_critics = (
        "<b>Redes Diana Estabilizadas: Q<sub>&theta;&macr;1</sub>(s', a') &amp; Q<sub>&theta;&macr;2</sub>(s', a')</b><br>"
        "• Arquitectura idéntica, parámetros congelados durante el paso hacia adelante<br>"
        "• Evaluación en s' con acción re-muestreada a' ~ &pi;<sub>&phi;</sub>(&middot;|s')<br>"
        "• Valor diana conservador: <b>min(Q<sub>&theta;&macr;1</sub>, Q<sub>&theta;&macr;2</sub>)</b> (Clipped Double Q)"
    )
    add_node(3104, node_target_critics, node_style("#E0E7FF", "#4F46E5", bold=True), 570, 760, 480, 65)

    # 3.4 Polyak Soft Update
    node_polyak = (
        "<b>Actualización Suave de Parámetros Diana (Polyak Averaging)</b><br>"
        "&theta;&macr;<sub>i</sub> &larr; (1 - &tau;<sub>polyak</sub>) &theta;&macr;<sub>i</sub> + &tau;<sub>polyak</sub> &theta;<sub>i</sub>, &nbsp;&nbsp;&nbsp;&tau;<sub>polyak</sub> = 0.005<br>"
        "<span style='font-size:10px;color:#1E40AF;'>Garantiza convergencia asintótica y erradica divergencias por sobreestimación de valor</span>"
    )
    add_node(3105, node_polyak, node_style("#FEF2F2", "#DC2626"), 570, 845, 480, 50)

    # 3.5 Critic Consensus Output
    node_consensus = (
        "<b>Evaluación del Actor: Q<sub>min</sub>(s, a)</b><br>"
        "Q<sub>min</sub> = min(Q<sub>&theta;1</sub>(s, &atilde;(s)), Q<sub>&theta;2</sub>(s, &atilde;(s))) &rarr; Utilizado para actualizar al Actor"
    )
    add_node(3106, node_consensus, node_style("#EFF6FF", "#1D4ED8", bold=True), 570, 915, 480, 45)

    # Internal Edges Critics
    add_edge(3201, "", edge_style("#2563EB"), 3101, 3102)
    add_edge(3202, "", edge_style("#2563EB"), 3101, 3103)
    add_edge(3203, "Evaluación s', a'", edge_style("#4F46E5"), 3104, 3105)
    add_edge(3204, "Gradientes de &theta;", edge_style("#DC2626", dashed=1), 3102, 3105)
    add_edge(3205, "", edge_style("#2563EB"), 3102, 3106)
    add_edge(3206, "", edge_style("#2563EB"), 3103, 3106)

    # =========================================================================
    # 4. SUBSYSTEM: BUFFER DE EXPERIENCIAS & BATCHING EN GPU
    # Bounds: x=1110, y=100, w=570, h=380
    # =========================================================================
    add_node(
        4000,
        "<b>SUBSISTEMA 4: EXPERIENCIA Y MUESTREO EN GPU (REPLAY BUFFER)</b><br>"
        "<span style='font-size:10px;font-weight:normal;color:#C2410C;'>ReplayBuffer &middot; Manejo de Lotes Multi-Entorno</span>",
        container_style("#FFF7ED", "#EA580C", "#C2410C"),
        1110, 100, 570, 380
    )

    # 4.1 Ingestion Multi-Env
    node_ingest = (
        "<b>Ingesta Vectorial en Paralelo: add_batch()</b><br>"
        "• Recibe transiciones de N<sub>envs</sub> = 4 entornos de simulación simultáneos<br>"
        "• Formas de Tensores: obs (B, 22), act (B, 3), rew (B, 1), next_obs (B, 22), done (B, 1)<br>"
        "• Inserción inmediata sin contención de hilos: 4 transiciones por paso RL"
    )
    add_node(4101, node_ingest, node_style("#FFEDD5", "#F97316", bold=True), 1130, 155, 530, 65)

    # 4.2 Preallocated Memory Pools
    node_ring_buffer = (
        "<b>Buffer Circular Pre-Asignado en Memoria (Capacidad = 50.000)</b><br>"
        "• <tt>obs_buf</tt>: np.zeros((50000, 22), dtype=float32) [~4.4 MB]<br>"
        "• <tt>action_buf</tt>: np.zeros((50000, 3), dtype=float32) [~0.6 MB]<br>"
        "• <tt>reward_buf</tt>: np.zeros((50000, 1), dtype=float32)<br>"
        "• <tt>next_obs_buf</tt>: np.zeros((50000, 22), dtype=float32)<br>"
        "• <tt>done_buf</tt>: np.zeros((50000, 1), dtype=float32)<br>"
        "• <i>Cero fragmentación de memoria RAM/VRAM en sesiones largas de 1000+ episodios</i>"
    )
    add_node(4102, node_ring_buffer, node_style("#FFFFFF", "#EA580C"), 1130, 235, 530, 95)

    # 4.3 GPU Sampler
    node_sampler = (
        "<b>Muestreador de Mini-Lotes en GPU: buffer.sample(batch_size=64/256)</b><br>"
        "• Índices aleatorios uniformes sobre buffer activo [0, size)<br>"
        "• Conversión y transferencia directa a device='cuda':<br>"
        "  <tt>torch.as_tensor(..., dtype=torch.float32, device='cuda')</tt><br>"
        "• Salida: Diccionario de Tensores PyTorch listo para backprop"
    )
    add_node(4103, node_sampler, node_style("#FEF3C7", "#D97706", bold=True), 1130, 345, 530, 65)

    add_edge(4201, "Push 4 envs", edge_style("#EA580C"), 4101, 4102)
    add_edge(4202, "Muestreo aleatorio", edge_style("#D97706"), 4102, 4103)

    # =========================================================================
    # 5. SUBSYSTEM: PÉRDIDAS, OPTIMIZACIÓN Y ENTROPÍA ADAPTATIVA
    # Bounds: x=1110, y=500, w=570, h=520
    # =========================================================================
    add_node(
        5000,
        "<b>SUBSISTEMA 5: ACTUALIZACIONES DE BELLMAN & ENTROPÍA ADAPTATIVA</b><br>"
        "<span style='font-size:10px;font-weight:normal;color:#991B1B;'>Loss Functions & Adam Optimizers (LR=3e-4)</span>",
        container_style("#FEF2F2", "#DC2626", "#991B1B"),
        1110, 500, 570, 520
    )

    # 5.1 Bellman Target Formulation
    node_bellman = (
        "<b>1. Objetivo de Bellman Suave (Target Q con Temperatura &alpha;)</b><br>"
        "<b>y = r + &gamma; (1 - done) &middot; [ min(Q<sub>&theta;&macr;1</sub>(s', a'), Q<sub>&theta;&macr;2</sub>(s', a')) - &alpha; log &pi;(a'|s') ]</b><br>"
        "• &gamma; = 0.99 (factor de descuento temporal)<br>"
        "• a' ~ &pi;<sub>&phi;</sub>(&middot;|s') muestreado con la política actual<br>"
        "• Evaluado en bloque <tt>with torch.no_grad():</tt> para aislar gradientes"
    )
    add_node(5101, node_bellman, node_style("#FFFFFF", "#DC2626", bold=True), 1130, 555, 530, 75)

    # 5.2 Critic Loss
    node_loss_q = (
        "<b>2. Pérdida del Crítico (&Lscr;<sub>Q</sub>) &rarr; Optimizador Adam (&theta;<sub>1</sub>, &theta;<sub>2</sub>)</b><br>"
        "<b>&Lscr;<sub>Q</sub>(&theta;<sub>1</sub>, &theta;<sub>2</sub>) = MSE(Q<sub>&theta;1</sub>(s, a), y) + MSE(Q<sub>&theta;2</sub>(s, a), y)</b><br>"
        "• Minimiza el error de evaluación de estado-acción cuadrático<br>"
        "• Actualización: critic_loss.backward() &rarr; critic_opt.step()"
    )
    add_node(5102, node_loss_q, node_style("#FEE2E2", "#B91C1C"), 1130, 645, 530, 65)

    # 5.3 Actor Loss
    node_loss_pi = (
        "<b>3. Pérdida del Actor (&Lscr;<sub>&pi;</sub>) &rarr; Optimizador Adam (&phi;)</b><br>"
        "<b>&Lscr;<sub>&pi;</sub>(&phi;) = E<sub>s~D, &epsilon;~&Nu;</sub> [ &alpha; log &pi;<sub>&phi;</sub>(a<sub>&phi;</sub>(s)|s) - min(Q<sub>&theta;1</sub>(s, a<sub>&phi;</sub>(s)), Q<sub>&theta;2</sub>(s, a<sub>&phi;</sub>(s))) ]</b><br>"
        "• Maximiza simultáneamente el valor Q esperado y la entropía de la acción<br>"
        "• Retropropaga a través de la reparametrización sin afectar a los críticos"
    )
    add_node(5103, node_loss_pi, node_style("#F3E8FF", "#7C3AED", bold=True), 1130, 725, 530, 75)

    # 5.4 Dual Temperature Loss (Alpha)
    node_loss_alpha = (
        "<b>4. Entropía Adaptativa Automática (&Lscr;<sub>&alpha;</sub>) &rarr; Optimizador log_alpha</b><br>"
        "<b>&Lscr;(&alpha;) = - E [ log &alpha; &middot; (log &pi;<sub>&phi;</sub>(a|s) + H&#773;) ]</b><br>"
        "• Entropía Objetivo Fija: <b>H&#773; = - dim(Acciones) = -3.0</b><br>"
        "• Parámetro positivo: &alpha; = exp(log_alpha) &gt; 0<br>"
        "• Autorregulación: si la entropía decae por debajo de -3.0, &alpha; aumenta (más exploración); si es alta, &alpha; disminuye"
    )
    add_node(5104, node_loss_alpha, node_style("#FEF3C7", "#D97706", bold=True), 1130, 815, 530, 85)

    # 5.5 Loss Metrics Output
    node_loss_out = (
        "<b>Métricas de Entrenamiento Exportadas (Logs JSONL / CSV)</b><br>"
        "• <tt>critic_loss</tt> (convergencia Bellman) | <tt>actor_loss</tt> (mejora de política)<br>"
        "• <tt>alpha</tt> (coeficiente de temperatura) | <tt>time_sec</tt> &amp; <tt>ep_reward</tt>"
    )
    add_node(5105, node_loss_out, node_style("#F1F5F9", "#475569"), 1130, 915, 530, 50)

    # Internal Edges Losses
    add_edge(5201, "Target Q (y)", edge_style("#DC2626"), 5101, 5102)
    add_edge(5202, "Min Q_curr", edge_style("#7C3AED"), 5102, 5103)
    add_edge(5203, "curr_log_prob", edge_style("#D97706"), 5103, 5104)
    add_edge(5204, "Registros", edge_style("#475569"), 5104, 5105)

    # =========================================================================
    # 6. SUBSYSTEM: GESTOR DE CURRÍCULO Y CONTROL MULTI-ETAPA (FSM)
    # Bounds: x=40, y=1040, w=1640, h=260
    # =========================================================================
    add_node(
        6000,
        "<b>SUBSISTEMA 6: GESTOR DE CURRÍCULO ADAPTATIVO & FSM MULTI-ETAPA</b><br>"
        "<span style='font-size:10px;font-weight:normal;color:#14532D;'>CurriculumManager &middot; Control de Fricción de Mesa y Promoción de Fases</span>",
        container_style("#F0FDF4", "#16A34A", "#14532D"),
        40, 1040, 1640, 260
    )

    # 6.1 Rolling Success Window
    node_curric_sr = (
        "<b>Ventana Deslizante de Éxito</b><br>"
        "• Tamaño: W = 15 episodios<br>"
        "• Criterio de Éxito de Episodio:<br>"
        "  <b>d<sub>final</sub> &le; &delta;<sub>threshold</sub> (0.05 m)</b><br>"
        "• Tasa de Éxito: SR = (N<sub>éxitos</sub> / W) &times; 100%<br>"
        "• <b>Umbral de Promoción: SR &ge; 85%</b>"
    )
    add_node(6101, node_curric_sr, node_style("#DCFCE7", "#16A34A", bold=True), 60, 1095, 300, 105)

    # 6.2 Table Friction Decay (Translating Penalty Spring k)
    node_curric_fric = (
        "<b>Decaimiento de Fricción de Mesa (&mu;<sub>table</sub>)</b><br>"
        "• <b>&mu; = 80.0 (Fase Inicial 'Fácil')</b>: Retención cuasi-infinita, prenda anclada, Bézier nominal domina<br>"
        "• <b>&mu; = 20.0 (Fase Media)</b>: Comienzo de deslizamientos inerciales y fuerzas tangenciales de arrastre<br>"
        "• <b>&mu; = 5.0 (Fase Final 'Difícil')</b>: Mesa deslizante; Bézier nominal falla (SR &lt; 50%), SAC residual corrige activamente en tiempo real (SR &ge; 90%)"
    )
    add_node(6102, node_curric_fric, node_style("#FEF3C7", "#D97706", bold=True), 380, 1095, 470, 105)

    # 6.3 Staged Folding FSM
    node_curric_stages = (
        "<b>FSM de Doblado Canónico en 3 Etapas Secuenciales</b><br>"
        "• <b>Etapa 1: Plegado Lateral de Pernera</b> &rarr; P<sub>0</sub>=l_corner &rarr; P<sub>target</sub>=r_corner (H=20 pasos, &Delta;x residual)<br>"
        "• <b>Etapa 2: Tracción Bicapa de Cintura</b> &rarr; P<sub>0</sub>=[top_ctr_f, top_ctr_b] (elevación 32 cm + sacudida anti-arrugas)<br>"
        "• <b>Etapa 3: Doblado Longitudinal Completo</b> &rarr; P<sub>0</sub>=[l_leg_o, r_leg_o] (tobillos) &rarr; P<sub>target</sub>=top_ctr_f (cintura)<br>"
        "• Transferencia de Pesos: Carga checkpoints canónicos automáticos <tt>sac_best_stage1.pt</tt>"
    )
    add_node(6103, node_curric_stages, node_style("#EDE9FE", "#7C3AED", bold=True), 870, 1095, 520, 105)

    # 6.4 Sim-to-Real Benchmark Suite
    node_eval_suite = (
        "<b>Suite de Evaluación y Renderizado HD</b><br>"
        "• Script: <tt>03_eval_policy.py</tt><br>"
        "• Intervalos Wilson Score 95%<br>"
        "• Video MP4 HD sin fugas VRAM<br>"
        "• Comparación vs Baseline Nominal"
    )
    add_node(6104, node_eval_suite, node_style("#F1F5F9", "#475569"), 1410, 1095, 250, 105)

    # Internal Edges Curriculum
    add_edge(6201, "SR &ge; 85%", edge_style("#16A34A"), 6101, 6102)
    add_edge(6202, "&mu; &le; 5.0 superado", edge_style("#7C3AED"), 6102, 6103)
    add_edge(6203, "Eval Stage", edge_style("#475569"), 6103, 6104)

    # =========================================================================
    # GLOBAL INTER-SUBSYSTEM CONNECTORS (THE CLOSED-LOOP INTERACTION)
    # =========================================================================
    # 1. Observation Flow: Environment (Subsystem 1) -> Actor (Subsystem 2) & Critic (Subsystem 3)
    add_edge(
        9001, "s_t (22D)",
        edge_style("#0D9488", width=3),
        1103, 2101,
        waypoints=[(500, 437), (500, 172)]
    )
    add_edge(
        9002, "s_t (22D)",
        edge_style("#0D9488", width=2),
        1103, 3101,
        waypoints=[(510, 437), (510, 632)]
    )

    # 2. Action Flow: Actor (Subsystem 2) -> Residual Blender (Subsystem 1)
    add_edge(
        9003, "Acción residual a_t (3D)",
        edge_style("#EF4444", width=3),
        2106, 1105,
        waypoints=[(540, 442), (540, 665)]
    )

    # 3. Action to Critic: Actor -> Critic
    add_edge(
        9004, "a_t (3D)",
        edge_style("#9333EA", width=2, dashed=1),
        2106, 3101,
        waypoints=[(810, 485), (810, 580), (810, 615)]
    )

    # 4. Environment Step -> Replay Buffer: (s, a, r, s', done)
    add_edge(
        9005, "add_batch(s_t, a_t, r_t, s_t+1, d_t) [4 envs]",
        edge_style("#EA580C", width=3),
        1107, 4101,
        waypoints=[(300, 920), (300, 950), (1090, 950), (1090, 187)]
    )

    # 5. Buffer -> Loss Computation: Sample Batch
    add_edge(
        9006, "Muestreo Mini-Lote GPU (Batch B=64/256)",
        edge_style("#D97706", width=3),
        4103, 5101,
        waypoints=[(1395, 420), (1395, 545)]
    )

    # 6. Gradient Flow from Losses to Actor & Critic
    add_edge(
        9007, "&nabla;_&phi; (Actualización Actor)",
        edge_style("#7C3AED", width=2, dashed=1, startArrow="classic", endArrow="none"),
        2102, 5103,
        waypoints=[(1060, 235), (1080, 235), (1080, 762)]
    )
    add_edge(
        9008, "&nabla;_&theta; (Actualización Críticos)",
        edge_style("#2563EB", width=2, dashed=1, startArrow="classic", endArrow="none"),
        3102, 5102,
        waypoints=[(1060, 702), (1085, 702), (1085, 677)]
    )

    # 7. Curriculum Feedback: Environment Metrics -> Curriculum Manager
    add_edge(
        9009, "Resultado Episodio (d_final, is_success)",
        edge_style("#16A34A", width=2),
        1107, 6101,
        waypoints=[(280, 920), (280, 1020), (210, 1020)]
    )

    # 8. Curriculum Control: Modulación de Fricción de Mesa -> Genesis Table
    add_edge(
        9010, "Ajuste coup_friction (&mu; = 80 &rarr; 20 &rarr; 5)",
        edge_style("#D97706", width=2, dashed=1),
        6102, 1101,
        waypoints=[(615, 1080), (615, 1030), (20, 1030), (20, 207), (60, 207)]
    )

    # Pretty print XML
    xml_str = ET.tostring(mxfile, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ')

    return pretty_xml


if __name__ == '__main__':
    xml_content = build_drawio_xml()
    
    # Save to genesis_ws Phase 5
    target_path_ws = Path("D:/genesis_ws/phase_5_gym_env/cloth_curriculum_rl/sac_architecture.drawio")
    target_path_ws.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path_ws, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"[OK] Diagrama Draw.io generado en: {target_path_ws}")

    # Save to TFM_Vault
    target_path_vault = Path("D:/TFM_Vault/03_AI_Training/sac_architecture_phase5.drawio")
    target_path_vault.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path_vault, "w", encoding="utf-8") as f:
        f.write(xml_content)
    print(f"[OK] Diagrama Draw.io respaldado en: {target_path_vault}")
