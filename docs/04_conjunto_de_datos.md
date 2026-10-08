# 4. Conjunto de datos

> Actualizado el 7 de octubre de 2026 sobre `rawdata/` (originales con nombres normalizados) y `personas_video.csv`. Las cifras de cobertura y propiedades salen de `evaluate_metrics.py` (`metricas/lstm_senha_model/`). La versión anterior del conjunto (`rawdata_clean/`, 79 originales en train, nombres `video_<fecha>` y ` - Trim`) ya no se usa.

## 4.1 Clases

10 señas de la LSPy, una carpeta por seña: agosto, broma, jugar, martes, marzo, miercoles, nombre, rojo, sabado, soltero. Con 10 clases equilibradas, **el azar acierta el 10 %** (línea base 1/K).

## 4.2 Cantidad de videos

| | Originales | Copias aumentadas | Total |
|---|---:|---:|---:|
| Train (7 a 9 por seña: martes 7, marzo 9, el resto 8) | 80 | 320 | 400 |
| Test (2 por seña) | 20 | 0 | 20 |
| **Total** | **100** | **320** | **420** |

- Cada original de train tiene **4 copias** (`_0` a `_3`), generadas el 7 de octubre con `data_augmentation.py --max-clips 4 --espejo` en `rawdata_aug/train/`: rotación (±30°), zoom (85 % – 115 %), eliminación de hasta el 20 % de los fotogramas, y espejo horizontal en las copias 0 y 2 (160 copias espejadas). Los parámetros de cada copia están en `rawdata_aug/train/aumento_parametros.csv`.
- En total hay **100 grabaciones reales**. El resto son variaciones de esas mismas grabaciones.
- `test/` tiene solo originales, sin copias: **la división en train y test se hizo antes del aumento**, que es lo correcto.
- La validación de la LSTM sale de train: el 20 % de los originales de cada seña, redondeado (2 por seña, 1 en martes): 19 originales, 95 secuencias con sus copias (ver 01 · 1.5).
- **Diferencia con la tesis:** el capítulo 6 habla de “50 videos por etiqueta (500 en total)” a partir de 10 originales con 4 copias. El conjunto tiene 8 originales por seña en train (7 en martes, 9 en marzo) y 2 en test: 40 secuencias de train por seña con las copias (35 en martes, 45 en marzo), más 2 de test. Ajustar el texto a estos números.
- `rawdata/trainaug/` contiene copias de una versión anterior del aumento; el pipeline actual no la usa.

## 4.3 Convención de nombres

| Patrón | Ejemplo | Significado |
|---|---|---|
| `<seña>_<persona>.mp4` | `agosto_cr5.mp4`, `agosto_sebastian.mp4` | Original |
| `<original>_<i>.mp4` | `agosto_cr5_2.mp4` | Copia aumentada (solo en train) |

**Caso especial:** `test/marzo/marzo_cr1_2.mp4` es un original, pero su nombre coincide con el de la copia aumentada `train/marzo/marzo_cr1_2.mp4` (copia 2 de `marzo_cr1`). El código lo resuelve: las secuencias se guardan por partición (`data/sequences/<partición>/`) y en test los nombres nunca se interpretan como copias (`lspy_common.video_original`). Aun así, renombrarlo (por ejemplo `marzo_cr1b.mp4`, actualizando `personas_video.csv`) evitaría confusiones.

## 4.4 Personas que signan y tipo de evaluación

`personas_video.csv` asigna una persona a cada original:

| Persona | Train | Test |
|---|---:|---:|
| cr3, cr4, cr6, cr7, cr8, marcelo, sebastian | 10 cada una (1 por seña) | — |
| cr5 | 9 (todas menos martes) | — |
| cr1 | 1 (`marzo_cr1`) | 9 |
| cr2 | — | 10 |
| cr1_2 | — | 1 (`marzo_cr1_2`) |

Una comparación visual de un fotograma por video coincide con el CSV: en cada seña, las personas de train son distintas entre sí y distintas de las de test, **con una excepción**: `test/marzo/marzo_cr1_2` y `train/marzo/marzo_cr1` muestran a la misma persona (misma ropa y fondo) haciendo la misma seña. Es decir, `cr1_2` parece ser `cr1`; convendría unificarlo en el CSV.

Conclusión: **la prueba es casi independiente del signante**. cr2 no aparece en train y cr1 aparece en un solo video de train. Solo 1 de los 20 videos de test (`marzo_cr1_2`) es de una persona que el modelo vio haciendo esa misma seña. Esto hace que los resultados sean más exigentes que una evaluación dependiente del signante y comparables con la situación “personas nuevas” del estado del arte (AUTSL, capítulo 3). Debe declararse en la metodología.

La validación interna de la LSTM, en cambio, sí es dependiente del signante: las mismas personas aparecen en el entrenamiento con otras señas.

Para analizar por persona:

```bash
python evaluate_metrics.py --sin-modelo --signers-csv personas_video.csv
python evaluate_metrics.py --model lstm_senha_model --cv 5 --cv-por-persona --cv-incluir-test --signers-csv personas_video.csv
```

## 4.5 Propiedades de los videos

| | Videos | 60 fps, 720 × 1280 | ≈ 30 fps, ≈ 360 × 650 | Duración (media; mín. – máx.) |
|---|---:|---:|---:|---|
| Train (originales) | 80 | 50 | 30 | 3,2 s; 1,7 – 8,5 s |
| Test | 20 | 20 | 0 | 3,7 s; 2,7 – 4,8 s |

Consecuencias:

- Los videos son **verticales** (9:16). `handtrack.py` los completa con bordes negros hasta hacerlos cuadrados antes de MediaPipe, sin deformarlos.
- Conviven videos de **60 fps y de 30 fps**, y de duraciones muy distintas. `handtrack.py` remuestrea cada video por tiempo a 150 fotogramas, así la seña completa ocupa siempre la secuencia entera. En los videos con menos de 150 cuadros (276 de 400 en train, sobre todo los de 30 fps) algunos cuadros se repiten: de ahí el 22 % de “pasos idénticos consecutivos” que informa `evaluate_metrics.py`.
- **Todo test es de 60 fps y alta resolución**, mientras que 30 de los 80 originales de train son de menor resolución y 30 fps. Es una diferencia de dominio entre train y test.
- MediaPipe detecta al menos una mano en el **99,4 %** de los fotogramas de train y el **99,8 %** de los de test (mínimo por video: 95 de 150).

## 4.6 Lo que conviene informar en la tesis sobre los datos

1. Número de personas que signan (10, más la duda sobre `cr1_2`) y cuántos videos aporta cada una (tabla 4.4).
2. Videos reales vs. aumentados por clase (tabla `tabla_dataset.tex` de `evaluate_metrics.py`).
3. Que la división train/test se hizo antes del aumento y que test contiene solo originales.
4. Que test es casi independiente del signante (solo `marzo_cr1_2` comparte persona y seña con train).
5. Resolución, fps y duración de los videos (`videos_crudos.csv`), y la diferencia de dominio entre train y test.
6. Cobertura de detección de manos de MediaPipe por clase (`tabla_cobertura_manos.tex`).
