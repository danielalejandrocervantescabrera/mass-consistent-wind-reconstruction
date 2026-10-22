# Ejemplo sintético temporal — implementación de referencia

**ACTUALIZACIÓN:** el protocolo experimental final fue recuperado y está en `final/`.
Consulte `final/README_ES.md` para reproducir las 16,384 observaciones y las tres semillas.

Este ejemplo representa un vórtice tridimensional cuyo centro sigue una trayectoria
parabólica. La entrada de la red es `(x,y,z,t)` y la salida es `(u,v,w)`.
Únicamente `(u,v)` se emplean en la pérdida de datos. La reconstrucción de `w`
se obtiene imponiendo aproximadamente la divergencia espacial nula y `w=0`
en la frontera inferior. El tiempo es una coordenada: no se realiza pronóstico.

## Estado de reproducibilidad (IMPORTANTE)

`src/time_dependent_vortex_reference.py` es una copia **sin alteraciones** del
programa facilitado por el autor. Su hash SHA-256 se registra en `SOURCE_SHA256.txt`.
La formulación del campo analítico y las condiciones físicas coinciden con el
artículo, pero **este archivo no reproduce el protocolo final de la sección 5.9**:

- Obtiene observaciones nuevas en cada lote, no 16,384 observaciones LHS fijas.
- No usa selección de checkpoint mediante MSE de validación horizontal.
- No implementa tres semillas explícitas ni la evaluación final independiente
  de 50,000 puntos descrita en el manuscrito.
- Sus parámetros por defecto y su ejemplo de invocación difieren del
  experimento publicado.

Por ello, los comandos de abajo son una **demostración funcional** y no deben
utilizarse para regenerar o sustituir las cifras o figuras publicadas.
La implementación final se encuentra ahora en `final/`; esta copia sigue siendo demostrativa.

## Comprobaciones rápidas

Desde la raíz del repositorio:

```bash
python3 synthetic/time_dependent/scripts/check_physics.py
bash synthetic/time_dependent/scripts/smoke_test.sh
```

El test de física verifica numéricamente la divergencia exacta cero y `w=0`
en la base. El test funcional entrena una época y genera archivos de ejemplo
que **no** corresponden a los resultados del artículo.

## Ejecutar configuración de demostración documentada en el programa

```bash
bash synthetic/time_dependent/scripts/run_reference.sh
```

Este comando es computacionalmente exigente (500 épocas, 256 lotes por época).
Para una prueba más corta:

```bash
EPOCHS=2 BATCHES=2 bash synthetic/time_dependent/scripts/run_reference.sh
```

Los archivos generados se escriben en `synthetic/time_dependent/outputs/`.

## Protocolo final descrito en el artículo (script localizado en `final/`)

Arquitectura residual ancho 60 / profundidad 2, 2,000 épocas, Adam 0.01,
25 lotes/época, 512 puntos interiores y 64 de frontera por lote;
N_obs=16,384 LHS fijas en espacio-tiempo. Selección de checkpoint por
validación de `(u,v)` y evaluación independiente final sobre 50,000 puntos.
Estos valores describen el artículo, **no** opciones ya verificadas de
este programa histórico.
