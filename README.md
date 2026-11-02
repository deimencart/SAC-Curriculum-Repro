# Soft Actor-Critic (SAC) & Curriculum Learning para Doblado Robótico de Ropa

> **Reproducción Académica e Implementación Experimental**  
> *Referencias: Haarnoja et al. (ICML 2018), Wang et al. (IEEE TPAMI 2021), Li et al. (2025)*  
> *Simulador: Genesis World (Física de Deformables XPBD / PBD en GPU)*  
> *Framework: PyTorch, Gymnasium, Control Continuo Residual en SE(3)*

---

> [!IMPORTANT]
> **Aviso de Reproducción Académica (Sin fines de plagio)**  
> Este repositorio contiene una **reproducción experimental e implementación académica independiente** realizada con fines educativos y de investigación en robótica y aprendizaje por refuerzo. Los modelos teóricos de Soft Actor-Critic y Curriculum Learning pertenecen y son atribuidos a sus autores originales (*Haarnoja et al., 2018; Wang et al., 2021; Li et al., 2025*). Esta implementación no tiene fines comerciales ni intención de plagio.

---

## 📌 ¿Qué problema resuelve este proyecto?

Entrenar a un manipulador robótico para doblar prendas flexibles utilizando Reinforcement Learning (RL) puro desde cero suele fracasar. Las telas tienen infinitos grados de libertad, se auto-colisionan de forma no lineal y el espacio de búsqueda es demasiado amplio para explorar a ciegas.

Para resolverlo, esta implementación combina dos conceptos fundamentales:
1. **Control Residual:** En lugar de dejar que la red controle el brazo desde cero, el robot sigue una trayectoria nominal suave (curvas de Bézier cúbicas $C^2$ de mínimo tirón). La política **Soft Actor-Critic (SAC)** solo aprende a aplicar correcciones residuales cartesianas finas ($\Delta \mathbf{x}$) para compensar arrugas y estiramientos.
2. **Curriculum Learning Multi-Etapa:** La dificultad de la tarea se modula progresivamente mediante una función de avance $\lambda(t) \in [0, 1]$. El robot empieza asistido en un rango estrecho y, conforme supera umbrales de éxito geométricos, se libera el control continuo total.

---

## 🛠️ ¿Qué implementamos en este repositorio?

El sistema integra un simulador físico de ropa en GPU con redes neuronales continuas en PyTorch:

```
[ Simulador Físico Genesis World ] (Física XPBD en GPU)
                 |
                 v
   [ Entorno Gymnasium (SE3) ]     (Pose del robot + Keypoints de tela)
                 |
                 v
   [ Controlador de Currículo ]    (Función de avance lambda(t) en 5 niveles)
                 |
                 v
   [ Red SAC (GaussianActor) ]     (Acciones continuas con squash Tanh)
                 |
                 v
 [ Acción Residual + Trayectoria ] (Movimiento nominal Bézier + Corrección Delta x)
```

### 1. Entorno de Simulación Gymnasium (`src/envs/`)
* **`dynamic_folding_env.py`:** Wrapper compatible con Gymnasium que conecta las acciones cartesianas con el brazo robótico y la pinza en Genesis World.
* **`cloth_genesis_backend.py`:** Interfaz de simulación en GPU que gestiona el motor XPBD, los parámetros elásticos de la tela y la detección de contacto.
* **`bezier_geometry.py`:** Generador de trayectorias analíticas de mínimo tirón (polinomios quínticos $C^2$) que garantizan movimientos suaves sin aceleraciones bruscas.
* **`measure_corner_fold.py`:** Módulo de métricas geométricas que calcula distancias Chamfer y errores de alineación entre perneras.

### 2. Algoritmo Soft Actor-Critic en PyTorch (`src/models/sac_policy.py`)
* **Actor Gaussiano:** Red neuronal que predice la media y desviación estándar de la corrección cartesiana, acotada suavemente mediante $\tanh(\cdot)$ con corrección analítica del Jacobiano para evitar divergencias.
* **Críticos Gemelos (Twin Critic):** Dos redes $Q$ independientes que aplican *Clipped Double-Q* para evitar la sobreestimación sistemática del valor.
* **Temperatura Adaptativa ($\alpha$):** Ajuste automático del balance entre explorar nuevas soluciones y explotar las acciones seguras.

### 3. Gestor de Currículo Automático (`src/callbacks/curriculum_callback.py`)
Controla el avance por niveles evaluando el desempeño en ventanas móviles:
* **Etapa 1 (Pliegue Lateral de Pernera):**
  * Nivel 1 ($\lambda = 0.0$): Trayectoria nominal pura asistida.
  * Niveles 2 a 4: Apertura gradual del rango de corrección residual ($\pm 2\,\text{cm}$ a $\pm 8\,\text{cm}$).
  * Nivel 5 ($\lambda = 1.0$): Control residual completamente libre.
* **Etapa 2 (Elevación Bicapa y Alisado por Gravedad):** Agarre bicapa de pretina (44 partículas), elevación vertical a $+65\,\text{cm}$ con batido transversal ("ondear sábana") y drapeado sobre la mesa alineando las botamangas ($\le 4\,\text{cm}$).

---

## 📊 Evidencias y Resultados de Entrenamiento

El entrenamiento converge a políticas robustas que superan los fallos de arrugado de los baselines analíticos:

| Curvas de Entrenamiento Etapa 1 (Escena Real) | Análisis Completo de 500 Episodios Etapa 2 |
| :---: | :---: |
| ![Stage 1 Curves](docs/images/training_curves_stage1_real_scene.png) | ![Stage 2 Analysis](docs/images/stage2_500ep_complete_training_analysis.png) |

| Evaluación de Nivel 5 (Acción Libre) | Progreso de Entrenamiento |
| :---: | :---: |
| ![Level 5 Eval](docs/images/eval_level5_free_500ep.png) | ![Progress](docs/images/training_progress.png) |

### Comparación en Vídeo
En la carpeta `videos/` se incluyen comparativas en formato MP4:
* `stage_1_nominal_baseline.mp4` vs. `stage_1_sac_best_attempt.mp4`: Corrección del pliegue de pernera.
* `stage_2_nominal_baseline.mp4` vs. `stage_2_sac_best_attempt.mp4`: Reducción del desalineamiento de tobillos de $14.2\,\text{cm}$ a solo $3.1\,\text{cm}$.

---

## 📁 Estructura del Repositorio

```text
sac_curriculum_cloth_folding/
├── README.md                                 <- Documentación general
├── .gitignore                                <- Exclusiones de Git
├── docs/
│   ├── sac_architecture.drawio              <- Diagrama de arquitectura de red
│   └── images/                               <- Gráficas de entrenamiento y convergencia
├── configs/
│   ├── training_params.yaml                  <- Hiperparámetros de SAC y del currículo
│   └── canonical_keypoint_poses.json         <- Posiciones canónicas de keypoints
├── src/
│   ├── callbacks/
│   │   └── curriculum_callback.py            <- Lógica de promoción curricular
│   ├── envs/
│   │   ├── dynamic_folding_env.py            <- Entorno Gymnasium
│   │   ├── cloth_genesis_backend.py          <- Interfaz de física XPBD Genesis
│   │   ├── bezier_geometry.py                <- Trayectorias Bézier C^2
│   │   └── measure_corner_fold.py            <- Evaluación de alineación y pliegue
│   ├── models/
│   │   └── sac_policy.py                     <- Implementación SAC en PyTorch
│   └── scripts/
│       ├── 01_verify_environment.py          <- Smoke test del simulador y GPU
│       ├── 02_train_sac_curriculum.py        <- Bucle de entrenamiento con currículo
│       ├── 03_eval_policy.py                 <- Evaluación de checkpoints guardados
│       └── record_policy_video.py            <- Grabación de MP4
├── checkpoints/                              <- Pesos entrenados (.pt) y trazas JSON
├── eval_reports/                             <- Informes de evaluación en JSON
└── videos/                                   <- Vídeos MP4 de evaluación
```

---

## 🚀 Guía Rápida de Ejecución

### 1. Dependencias
```bash
pip install torch torchvision gymnasium pyyaml matplotlib scipy
```
*Requiere tener instalado Genesis World (`genesis-world`).*

### 2. Verificar el Simulador y la GPU
```bash
python src/scripts/01_verify_environment.py
```

### 3. Entrenar la Política con Currículo
```bash
python src/scripts/02_train_sac_curriculum.py --config configs/training_params.yaml
```

### 4. Evaluar Checkpoints Entrenados
```bash
python src/scripts/03_eval_policy.py --checkpoint checkpoints/sac_best_stage1.pt
```

---

## 📚 Referencias
* Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML.
* Wang, X., et al. (2021). *A Survey on Curriculum Learning*. IEEE TPAMI.
* Li, et al. (2025). *Dynamic Cloth Folding with Curriculum*.
