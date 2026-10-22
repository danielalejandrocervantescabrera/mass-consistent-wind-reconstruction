# Ejemplo sintético temporal — protocolo final de reconstrucción

## Qué incluye

Se conservan **sin editar** los programas y `.sh` suministrados por el autor.
Los scripts nuevos en `scripts/` son lanzadores con rutas independientes del
lugar desde el que se ejecutan. La implementación utiliza datos fijos de
`(u,v)` en espacio-tiempo, divergencia espacial y condición inferior `w=0`;
**no pronostica** estados futuros. El caso real de Melissa extiende este
principio de reconstrucción con observaciones horarias procesadas.

## Configuración final

- Vórtice con trayectoria parabólica en `[-25,25]×[-25,25]×[0,7]×[-1,1]`.
- 16 384 observaciones LHS fijas, semilla 12345; no se usa `w` como dato.
- Red residual: ancho 60, profundidad 2; tres semillas 0, 1, 2.
- Adam, 2 000 épocas, 25 lotes/época, `lr=0.01`; 512 muestras interiores,
  64 en la frontera por lote; `beta1=beta2=1`.
- Selección de checkpoint **solo** por MSE de validación de `(u,v)`.
- Evaluación virgen con semilla 20260929: 50 000 puntos 4D y 20 000 puntos
  en la frontera; nueve cortes temporales de diagnóstico adicionales.
- Las desviaciones estándar entre las tres semillas se calculan con `ddof=1`.

## Cómo ejecutar

Desde la raíz del paquete:

```bash
bash synthetic/time_dependent/final/scripts/00_check.sh
bash synthetic/time_dependent/final/scripts/01_smoke.sh
# Entrenamiento completo (requiere tiempo de GPU):
bash synthetic/time_dependent/final/scripts/02_train_final.sh
# Evaluación de checkpoints congelados, sin entrenamiento:
bash synthetic/time_dependent/final/scripts/03_evaluate_virgin.sh
# Figuras y archivos para Gnuplot:
bash synthetic/time_dependent/final/scripts/04_figures.sh
# Barrido de cantidades de observaciones (por separado):
bash synthetic/time_dependent/final/scripts/05_obs_sweep.sh
```

Opcionalmente, `bash synthetic/time_dependent/final/scripts/06_all_final.sh`
realiza verificación de dependencias, entrenamiento, prueba virgen y figuras.
El *smoke test* usa una arquitectura pequeña y una sola época: **no produce
resultados publicables**.

Los entrenamientos nuevos se guardan en `outputs/new_training/`;
los checkpoints históricos se conservan bajo sus rutas originales.
La evaluación independiente genera los CSV en `outputs/evaluation_historical/`
o `outputs/evaluation_new/` según `TEMPORAL_CKPT_SET`.

## Límites y procedencia

- Los pesos originales de las tres redes y los CSV históricos específicos
  del experimento temporal final **ya están incluidos** desde la versión 4.
- El script de entrenamiento histórico reutilizaba opcionalmente la semilla 0
  de un barrido previo. Nuestro lanzador **evita** esa reutilización: las tres
  semillas se producen dentro de un mismo directorio.
- El barrido `run_time_obs_sweep.sh` utiliza una semilla de entrenamiento;
  debe interpretarse como estudio de sensibilidad, no como resultado de tres
  ejecuciones finales.
- La evaluación virgen utiliza la referencia analítica conocida **solo para
  medir los errores** del ejemplo sintético. No hay verdad terreno de `w`
  observada independientemente en el caso de Melissa.
- No se asegura reproducibilidad bit a bit entre versiones de PyTorch/CUDA.
- Un resultado publicado solo se confirma mediante los registros históricos;
  la prueba reducida demuestra que el programa funciona, no que reproduce
  las métricas numéricas publicadas.

Los archivos históricos originales se conservan. `legacy_evaluation/` alberga
un evaluador anterior que se suministra únicamente para procedencia.

## Figuras y exportación

`04_figures.sh` genera directamente figuras PDF/EPS mediante
`plot_time_final_16384.py` y tablas `.dat` para Gnuplot mediante
`export_time_final_16384_for_gnuplot.py`. Los archivos `.dat` no incluyen
el estilo Gnuplot histórico `.gp`, que no fue proporcionado; la figura PDF
sí se puede reconstruir con el programa Python incluido.


## Datos históricos recuperados (versión 4)

Se incorporaron los tres checkpoints originales correspondientes a semillas 0, 1 y 2:
la semilla 0 proviene de `out_time_obs_sweep`, semillas 1 y 2 de
`out_time_final_16384`. Se conservan **sin renombrar** las rutas y los CSV originales.
Sus configuraciones (16 384 LHS, `obsSeed=12345`, ResNet 60x2, 2 000 épocas,
penalizaciones unitarias, Adam/StepLR) concuerdan y la selección de checkpoints
se hizo con MSE horizontal de validación. Se adjunta SHA-256 de cada peso.

Revisar resúmenes históricos sin entrenar:

```bash
bash synthetic/time_dependent/final/scripts/07_verify_history.sh
```

Repetir la evaluación independiente **de los modelos históricos**:

```bash
bash synthetic/time_dependent/final/scripts/03_evaluate_virgin.sh
```

La evaluación genera los CSV en `outputs/evaluation_historical/`; los CSV
originales se conservan en `reference/independent/` para contrastarlos.
La evaluación puede tardar en CPU; no se ejecutó completa al empaquetar.

Para entrenar modelos nuevos, usar `02_train_final.sh`, y después

```bash
TEMPORAL_CKPT_SET=new bash synthetic/time_dependent/final/scripts/03_evaluate_virgin.sh
TEMPORAL_CKPT_SET=new bash synthetic/time_dependent/final/scripts/04_figures.sh
```

Los lanzadores crean directorios de ejecución aislados para no mezclar
los checkpoints históricos de `out_time_obs_sweep` con nuevas ejecuciones.
`06_all_final.sh` aplica por defecto el modo `new` después del entrenamiento.

Nota: se dispone de resúmenes y pesos históricos, pero este paquete no
contiene necesariamente los archivos originales completos de observaciones
ni los registros por época. No se afirma reproducción bit a bit en hardware distinto.
