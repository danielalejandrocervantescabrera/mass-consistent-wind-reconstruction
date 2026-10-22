# Reproducción de la reconstrucción del campo de viento

Organización reproducible de los programas **verificados** del artículo.
La red aproxima (u,v,w) a partir de datos horizontales (u,v), penalizando
la divergencia espacial y la condición de frontera inferior. **No realiza
pronósticos**. El experimento de Melissa sigue el ejemplo sintético temporal.

## Estructura

```text
melissa/           Caso real: datos originales, preprocesamiento, tres semillas,
                   análisis cinemático, estabilidad de w y figuras del artículo.
synthetic/example1/     Experimento sintético estacionario y densidad de observaciones.
synthetic/baselines/sasaki/  Método clásico comparativo y análisis TSVD.
synthetic/time_dependent/    Vórtice temporal: implementación de referencia y protocolo final
                             en `final/`, con LHS fijo, checkpoints y prueba virgen.
scripts/check_sources.sh    Auditoría de sintaxis e inventario de fuentes.
```

## Instalación (Linux/Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r melissa/requirements.txt
bash scripts/check_sources.sh
```

### Melissa (61 horas: reconstrucción)

```bash
cd melissa
bash scripts/00_preflight.sh
bash scripts/01_prepare.sh
bash scripts/02_train.sh
bash scripts/03_analyze.sh
bash scripts/04_figures.sh
bash scripts/04_report.sh
```

Usa semillas 1234, 2345 y 3456, 2500 épocas, observaciones completas,
Wahba lambda=0.001, interpolación vertical lineal, PCHIP temporal a 5 min.
Los datos originales se incluyen. `run_all.sh` realiza estas etapas. Los pesos
históricos de Melissa **están incluidos** bajo `melissa/reference/checkpoints/`.
Para verificar y analizar sin volver a entrenar, ejecutar desde la raíz:
`bash melissa/scripts/06_verify_history.sh` y después
`bash melissa/scripts/07_recompute_historical.sh`.

Prueba funcional reducida:

```bash
bash scripts/smoke_test.sh
```

### Ejemplo sintético dependiente del tiempo

El protocolo final se encuentra ahora en `synthetic/time_dependent/final/`:

```bash
bash synthetic/time_dependent/final/scripts/00_check.sh
bash synthetic/time_dependent/final/scripts/01_smoke.sh
# Entrenar tres semillas con 16,384 observaciones LHS fijas:
bash synthetic/time_dependent/final/scripts/02_train_final.sh
# Evaluación independiente de 50,000 puntos, sin nuevo entrenamiento:
bash synthetic/time_dependent/final/scripts/03_evaluate_virgin.sh
# Figuras finales:
bash synthetic/time_dependent/final/scripts/04_figures.sh
```

La versión previa de referencia y demostración se conserva separada. Los pesos
originales de las tres redes temporales **están incluidos**. La evaluación
virgen sobre 50,000 puntos se reprodujo usando esos tres checkpoints. Para
cotejar metadatos y CSV históricos sin entrenar:
`bash synthetic/time_dependent/final/scripts/07_verify_history.sh`.

### Ejemplo sintético I y Sasaki

```bash
cd ../synthetic/example1
bash smoke_test.sh
DEVICE=cuda:0 bash run_final.sh
# Ejecutar el estudio completo de densidad y la prueba de 50 000 puntos:
DEVICE=cuda:0 bash run_density_and_final_test.sh
```

Desde la raíz del paquete:

```bash
bash synthetic/baselines/sasaki/run_example2.sh
```

## Evidencia y límites

`reference/` contiene resultados históricos del artículo; `outputs/` registra
salidas nuevas. La ejecución de los scripts originales puede variar con el
hardware y la librería numérica. Se incluyen los pesos históricos para Melissa
y el ejemplo temporal, así como sus resúmenes; las respuestas HTTP originales
del servicio meteorológico y el registro primario de la trayectoria del huracán
no están disponibles. La evaluación independiente temporal se recalculó a partir
de los checkpoints y coincidió con los valores históricos. Para Melissa, la
integridad y arquitectura de los modelos fueron verificadas; sus indicadores
físicos se deben recalcular mediante `melissa/scripts/07_recompute_historical.sh`.

Las comparaciones cinemáticas y la variabilidad entre semillas examinan la
consistencia de la componente vertical reconstruida, no una validación con
mediciones independientes de velocidad vertical.

## Registro de entorno

Para documentar versiones reales, CUDA y hardware, ejecutar antes de entrenar:

```bash
bash scripts/record_environment.sh
```

La reproducibilidad bit a bit no está garantizada por los requisitos mínimos
del paquete. El archivo `outputs/environment.txt` permite registrar el entorno.

**Nota de procedencia:** el script del ejemplo sintético estacionario corresponde
a la versión disponible entre los archivos compartidos. El corredor histórico
de evaluación admite también otra variante si esta existía en el directorio
original; no se ha demostrado aún qué variante concreta se usó para cada
resultado antiguo. Se conservan referencias históricas separadas.


**Archivos históricos del ejemplo temporal.** Se añadieron los tres pesos y los CSV históricos de
evaluación independiente del ejemplo temporal; ejecutar
`bash synthetic/time_dependent/final/scripts/07_verify_history.sh` para auditar.
No es necesario reentrenar para examinar las métricas publicadas.


## Melissa: pesos originales recuperados (v5)

Se archivaron los tres modelos originales de Melissa (1234, 2345, 3456) con hashes SHA-256, sin mezclarlos con entrenamientos nuevos. Para verificar e inspeccionar sus métricas guardadas: `bash melissa/scripts/06_verify_history.sh`. Para recalcular los diagnósticos sin entrenar: `bash melissa/scripts/07_recompute_historical.sh`. Para las figuras: `bash melissa/scripts/08_figures_historical.sh`. Véase `melissa/README_ES.md` para requisitos y límites científicos.

## Publicación en GitHub y recálculo de ajuste horizontal

Consulta `docs/PUBLISHING.md` para subir esta versión como repositorio nuevo, preferentemente privado hasta revisar la atribución y licencia. El archivo `melissa/scripts/evaluate_melissa_uv_historico.py` permite volver a calcular las métricas horizontales desde los tres modelos originales sin entrenar:

```bash
bash melissa/scripts/09_recompute_uv_historical.sh
```

El ajuste horizontal sobre las 61 horas utilizadas para entrenar fue reproducido. Esto **no** equivale a validación temporal independiente ni a validación de `w` mediante mediciones independientes.
