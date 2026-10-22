# Melissa — paquete reproducible (experimento final)

Este paquete reúne los **cinco programas esenciales**, los **61 archivos horarios originales** y los resúmenes publicados. La tercera componente `w` se reconstruye a partir de `u,v`, divergencia espacial y condición de frontera inferior; **no hacemos predicción ni pronóstico**. Sigue la formulación del ejemplo sintético dependiente del tiempo.

## Instalación (Ubuntu/Linux)

```bash
cd melissa
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/00_preflight.sh
```

Para utilizar CUDA, instala el PyTorch compatible con tu GPU y controladores.

## Experimento completo

```bash
bash scripts/01_prepare.sh  # 61 horas, Wahba lambda=0.001, malla 40x40x15, PCHIP 5 min
bash scripts/02_train.sh    # 3 semillas 1234, 2345 y 3456: 2500 épocas, 25 lotes/época
bash scripts/03_analyze.sh  # Consistencia cinemática y dispersión entre semillas
bash scripts/04_report.sh   # Tabla de promedios y desviaciones muestrales
```

También puedes ejecutar todo el protocolo final con `bash scripts/run_all.sh` (incluye los tres entrenamientos completos).

Una prueba funcional reducida (NO reproduce los valores del artículo):

```bash
bash scripts/smoke_test.sh
```

Los resultados nuevos se guardan en `outputs/`. La carpeta `reference/` contiene **solo los resúmenes originales publicados**, para comprobar cifras sin repetir entrenamiento; no son una nueva ejecución.

**Actualización de la versión 5:** se incluyeron los pesos históricos originales `final_model.pt` de las tres semillas 1234, 2345 y 3456 en `reference/checkpoints/`. Cada uno tiene un hash SHA-256. Los resúmenes archivados son idénticos a los proporcionados junto con los pesos. Los arreglos procesados `train_X.npy/train_Yuv.npy` **no se incluyen**: se regeneran a partir de las 61 observaciones horarias en `data/raw/`.

El CSV de centros conserva una antigua marca `excluded=1` en 20251028T12, pero el preprocesamiento de la versión final **utiliza las 61 horas** y no aplica esa marca.

El descargador original (`src/download_melissa_power_hourly.py`) se incluye para transparencia. **No es necesario volver a descargar nada**. Su ejecución requiere autorización explícita:

```bash
ALLOW_DOWNLOAD=1 bash scripts/05_download_OPTIONAL.sh
```

Esto puede realizar hasta 48,800 solicitudes HTTP si faltaran todos los datos. La descarga usa WSC (velocidad corregida en altura), WD10M (dirección de 10 m) y `wind-surface=openwater`. Los perfiles de viento no son mediciones vectoriales independientes en cada altura.

## Archivos y límites de reproducibilidad

- `src/`: descarga, preparación, entrenamiento, diagnósticos verticales, estabilidad entre semillas.
- `data/raw/`: los 61 campos originales (48,800 filas) y trayectoria horaria.
- `data/processed/`: se genera; 17,304,000 muestras procesadas (salida de la malla temporal).
- `outputs/models/`: se genera `final_model.pt` y `summary.json` por semilla.
- `outputs/analysis/`: se generan figuras y métricas derivadas.
- `reference/`: resúmenes verificados y dos figuras compuestas del artículo, para cotejo documental.

No se recibieron los registros HTTP originales ni el documento primario del NHC para los 12 centros de referencia. El programa original `make_melissa_paper_figures.py` se recuperó posteriormente y está incluido; véase `scripts/08_figures_historical.sh` para las figuras compuestas. Consultar `scripts/04_report.sh` para verificar las cifras archivadas.


## Figuras compuestas del manuscrito

El programa ORIGINAL `src/make_melissa_paper_figures.py` fue recuperado del proyecto y se incluye
sin reescribir su contenido. Después de obtener `final_model.pt` para las tres semillas,
ejecuta:

```bash
bash scripts/03_analyze.sh
bash scripts/04_figures.sh
```

La segunda instrucción utiliza `outputs/analysis/seed_stability/w_fields_all_real_times.npz`
y el modelo `seed_2345/final_model.pt`. Produce cuatro figuras (PDF/PNG) y
`melissa_peak_figure_data.npz` en `outputs/figures/`.

Las figuras archivadas en `reference/` sirven para comparar, no prueban por sí mismas
que los pesos de una ejecución nueva sean idénticos a los originales. La reproducción
numérica puede depender de versiones de PyTorch, GPU y determinismo de CUDA.

## Modelos históricos: ejecución sin entrenar (v5)

Desde la **raíz del paquete completo** `mass-consistent-wind-reconstruction/` y con el entorno virtual activado:

```bash
bash melissa/scripts/06_verify_history.sh
bash melissa/scripts/07_recompute_historical.sh
bash melissa/scripts/08_figures_historical.sh
```

- `06` comprueba los 61 archivos de entrada, la integridad SHA-256, que los tres *state dictionaries* carguen con `weights_only=True` y la coherencia de su arquitectura y resúmenes. Los indicadores impresos proceden de `summary.json`: **no son un nuevo cálculo**.
- `07` prepara los datos a resolución final, si faltan; luego evalúa el modelo representativo (semilla 2345) para la consistencia cinemática y los tres modelos para la dispersión entre semillas. **No entrena ni modifica los pesos archivados.** El preprocesamiento de 17,304,000 puntos puede consumir memoria y espacio temporal significativos.
- `08` recrea las figuras compuestas usando los resultados de `07`.

Las salidas de esta vía quedan en `melissa/outputs/historical_analysis/` y `melissa/outputs/historical_figures/`. Los modelos originales permanecen en `melissa/reference/checkpoints/` y no están sujetos a los scripts de entrenamiento (`02_train.sh`). Las comparaciones cinemáticas de `w` comprueban coherencia física frente a un incremento deducido de `u,v`, **no son validación con mediciones independientes de velocidad vertical**. Las series de PCHIP tampoco son horas observadas independientes; no se ha realizado una validación temporal tipo leave-one-hour-out.

Para entrenamientos **nuevos** (opcional), se mantiene `bash melissa/scripts/02_train.sh`; estos se guardan en `melissa/outputs/models/` y nunca se usan por defecto en `06`–`08`.

## Recálculo independiente del ajuste horizontal (61 horas asimiladas)

Desde la raíz del repositorio, después de preparar los datos:

```bash
bash melissa/scripts/09_recompute_uv_historical.sh
```

El programa carga los tres pesos originales (1234/2345/3456), calcula los errores horizontales sobre las 61 horas usadas en entrenamiento y los compara con los `summary.json` archivados. En Ubuntu con CUDA se reproducen `5.77781044 ± 1.11904373 %` de error relativo y `0.78146723 ± 0.15135422 m/s` de RMSE (media ± desviación muestral). **No es un conjunto independiente de validación.**
