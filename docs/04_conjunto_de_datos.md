# 4. Conjunto de datos

> Actualizado el 8 de octubre de 2026 sobre `rawdata/` y `personas_video.csv`, después de equilibrar el conjunto: `marzo_cr1` pasó de train a test (en lugar de `marzo_cr1_2`, que era la misma persona) y se agregó `martes_cr5` en train. Los resultados obtenidos con la versión del 7 de octubre se respaldaron en `../respaldo_dataset_v2_2026-10-07/`. Las cifras de cobertura y propiedades salen de `evaluate_metrics.py` (`metricas/lstm_senha_model/`). La versión anterior del conjunto (`rawdata_clean/`, 79 originales en train, nombres `video_<fecha>` y ` - Trim`) ya no se usa.

## 4.1 Clases

10 señas de la LSPy, una carpeta por seña: agosto, broma, jugar, martes, marzo, miercoles, nombre, rojo, sabado, soltero. Con 10 clases equilibradas, **el azar acierta el 10 %** (línea base 1/K).

## 4.2 Cantidad de videos

| | Originales | Copias aumentadas | Total |
|---|---:|---:|---:|
| Train (8 por seña) | 80 | 320 | 400 |
| Test (2 por seña) | 20 | 0 | 20 |
| **Total** | **100** | **320** | **420** |

- Cada original de train tiene **4 copias** (`_0` a `_3`), generadas el 8 de octubre con `data_augmentation.py --max-clips 4 --espejo` en `rawdata_aug/train/`: rotación (±30°), zoom (85 % – 115 %), eliminación de hasta el 20 % de los fotogramas, y espejo horizontal en las copias 0 y 2 (160 copias espejadas). Los parámetros de cada copia están en `rawdata_aug/train/aumento_parametros.csv`.
- En total hay **100 grabaciones reales**. El resto son variaciones de esas mismas grabaciones.
- `test/` tiene solo originales, sin copias: **la división en train y test se hizo antes del aumento**, que es lo correcto.
- La validación de la LSTM sale de train: el 20 % de los originales de cada seña, redondeado (2 por seña): 20 originales, 100 secuencias con sus copias (ver 01 · 1.5).
- **Diferencia con la tesis:** el capítulo 6 habla de “50 videos por etiqueta (500 en total)” a partir de 10 originales con 4 copias. El conjunto tiene 10 originales por seña (uno por persona): 8 en train y 2 en test, es decir, 40 secuencias de train por seña con las copias, más 2 de test. Ajustar el texto a estos números.
- `rawdata/trainaug/` contiene copias de una versión anterior del aumento; el pipeline actual no la usa.

## 4.3 Convención de nombres

| Patrón | Ejemplo | Significado |
|---|---|---|
| `<seña>_<persona>.mp4` | `agosto_cr5.mp4`, `agosto_sebastian.mp4` | Original |
| `<original>_<i>.mp4` | `agosto_cr5_2.mp4` | Copia aumentada (solo en train) |

Desde el 8 de octubre ningún original de test termina en `_<dígito>`, así que los nombres ya no se confunden con copias aumentadas. De todos modos, el código guarda las secuencias por partición (`data/sequences/<partición>/`) y nunca interpreta los nombres de test como copias (`lspy_common.video_original`).

## 4.4 Personas que signan y tipo de evaluación

`personas_video.csv` asigna una persona a cada original. El conjunto está **equilibrado**: cada una de las 10 personas signa cada una de las 10 señas exactamente una vez.

| Persona | Train | Test |
|---|---:|---:|
|P3,P4,P5,P6,P7,P8, marcelo, sebastian | 10 cada una (1 por seña) | — |
|P1,P2 | — | 10 cada una (1 por seña) |

Conclusión: **la prueba es independiente del signante**. Ninguna persona de test aparece en train, así que los resultados miden la generalización a **personas nuevas**, la situación más exigente del estado del arte (AUTSL, capítulo 3). Debe declararse en la metodología.

La validación interna de la LSTM, en cambio, es dependiente del signante: las mismas personas aparecen en el entrenamiento con otras señas. La validación cruzada por persona (`--cv-por-persona`, 10 personas en 5 pliegues de 2) es independiente del signante, igual que la prueba.

Para analizar por persona:

```bash
python evaluate_metrics.py --sin-modelo --signers-csv personas_video.csv
python evaluate_metrics.py --model lstm_senha_model --cv 5 --cv-por-persona --cv-incluir-test --signers-csv personas_video.csv
```

## 4.5 Propiedades de los videos

| | Videos | 60 fps, 720 × 1280 | ≈ 30 fps, ≈ 360 × 650 | Duración (media; mín. – máx.) |
|---|---:|---:|---:|---|
| Train (originales) | 80 | 49 (+1 de 60 fps a 478 × 850) | 30 | 3,3 s; 1,7 – 8,5 s |
| Test | 20 | 20 | 0 | 3,7 s; 2,6 – 4,8 s |

Consecuencias:

- Los videos son **verticales** (9:16). `handtrack.py` los completa con bordes negros hasta hacerlos cuadrados antes de MediaPipe, sin deformarlos.
- Conviven videos de **60 fps y de 30 fps**, y de duraciones muy distintas. `handtrack.py` remuestrea cada video por tiempo a 150 fotogramas, así la seña completa ocupa siempre la secuencia entera. En los videos con menos de 150 cuadros (272 de 400 en train, sobre todo los de 30 fps) algunos cuadros se repiten: de ahí el 22 % de “pasos idénticos consecutivos” que informa `evaluate_metrics.py`.
- **Todo test es de 60 fps y alta resolución**, mientras que 30 de los 80 originales de train son de menor resolución y 30 fps. Es una diferencia de dominio entre train y test.
- MediaPipe detecta al menos una mano en el **99,4 %** de los fotogramas de train y el **99,8 %** de los de test (mínimo por video: 95 de 150).

## 4.6 Lo que conviene informar en la tesis sobre los datos

1. Número de personas que signan (10) y que cada una aporta un video de cada seña (tabla 4.4).
2. Videos reales vs. aumentados por clase (tabla `tabla_dataset.tex` de `evaluate_metrics.py`).
3. Que la división train/test se hizo antes del aumento y que test contiene solo originales.
4. Que test es independiente del signante (las 2 personas de test no aparecen en train).
5. Resolución, fps y duración de los videos (`videos_crudos.csv`), y la diferencia de dominio entre train y test.
6. Cobertura de detección de manos de MediaPipe por clase (`tabla_cobertura_manos.tex`).
