# Experimentos sintéticos

- `example1/`: reconstrucción estacionaria con observaciones LHS. Ejecutar
  `bash synthetic/example1/run_final.sh` para las tres semillas de la configuración
  fija; para el estudio de densidad y el conjunto final independiente, ejecutar
  `cd synthetic/example1 && DEVICE=cuda:0 bash run_density_and_final_test.sh`.
  La ejecución final de 50 000 puntos **no es** prueba prospectiva nueva: ya fue
  inspeccionada en el desarrollo original. Para reproducir números históricos hay
  que usar el mismo modelo y configuración registrados.
- `example1/reference/architecture_sensitivity`: 42 resúmenes CSV originales
  (14 configuraciones x 3 semillas), solo archivos de resultados, no sus modelos.
- `baselines/sasaki`: implementaciones originales de ejemplos I/II y prueba TSVD;
  ejecutar `bash synthetic/baselines/sasaki/run_example2.sh`. Los métodos
  comparativos no son intercambiables con la red.
- `time_dependent/`: documentación del ejemplo temporal; falta su script original.

**Importante:** los scripts de la red guardan resultados en `outputs/` y
ninguno reemplaza resultados de `reference/`. Se usan medias y desviaciones
muestrales (ddof=1) para tres semillas en las tablas revisadas.


## Experimento temporal final

El protocolo final verificado en código se encuentra en `time_dependent/final/`.
Ver `time_dependent/final/README_ES.md`.
