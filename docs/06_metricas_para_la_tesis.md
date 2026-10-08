# 6. Métricas para la tesis: `evaluate_metrics.py`

Script nuevo, en la raíz del repositorio, que genera todas las cifras, tablas LaTeX y figuras que el capítulo de resultados puede necesitar, a partir de lo que el pipeline ya produce. No modifica ningún archivo existente.

## 6.1 Instalación

Usa el mismo entorno del proyecto (`scikit-learn` ya está en `requirements.txt`):

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

(SciPy, Matplotlib y OpenCV ya están en `requirements.txt`.)

## 6.2 Modos de uso

```bash
# A) Solo datos y curvas de entrenamiento (no necesita el modelo ni TensorFlow para la auditoría)
python evaluate_metrics.py --sin-modelo --raw-dir rawdata_aug

# B) Evaluación completa del modelo final
python evaluate_metrics.py --model lstm_senha_model --raw-dir rawdata_aug

# C) Usar el mejor checkpoint (menor val_loss) en lugar del modelo de la última época
python evaluate_metrics.py --model mejor

# D) Todo: embeddings, latencia de Inception y validación cruzada de 5 pliegues
python evaluate_metrics.py --model lstm_senha_model --raw-dir rawdata_aug \
    --embeddings --latencia-inception \
    --cv 5 --cv-epochs 60 --cv-incluir-test --cv-solo-originales

# E) Validación cruzada dejando personas fuera
python evaluate_metrics.py --sin-modelo --cv 5 --cv-por-persona \
    --signers-csv personas_video.csv

# F) Comparar la arquitectura original con una liviana (misma partición de pliegues)
python evaluate_metrics.py --sin-modelo --cv 5 --cv-arch original --out-dir metricas/cv_original
python evaluate_metrics.py --sin-modelo --cv 5 --cv-arch ligera   --out-dir metricas/cv_ligera
```

### Opciones

| Opción | Por defecto | Descripción |
|---|---|---|
| `--data-dir` | `data` | Salida de `handtrack.py` |
| `--model` | `lstm_senha_model` | SavedModel, `.h5`/`.hdf5`/`.keras`, o `mejor` |
| `--sin-modelo` | — | Omite la evaluación del modelo |
| `--splits` | `test train` | Particiones a evaluar |
| `--seq-length`, `--max-frames`, `--class-limit`, `--data-type` | 150, 150, 10, (del `.json` del modelo) | Igual que en `DataSet`. `--data-type` se toma de `<modelo>.json`; si no existe, `features2048`. Valores: `features2048`, `probs`, `landmarks`, `features` (versión anterior) |
| `--logs-dir` / `--log-file` | `data/logs` / el más reciente | Log de `CSVLogger` |
| `--raw-dir` | — | Carpeta de videos para fps, resolución y duración |
| `--signers-csv` | — | CSV `video_original,persona[,particion,clase]` (p. ej. `personas_video.csv`); el nombre puede llevar extensión |
| `--out-dir` | `metricas/<fecha_hora>` | Carpeta de resultados |
| `--bootstrap` | 2000 | Remuestreos para los intervalos de confianza (0 = desactivar) |
| `--seed` | 42 | Semilla para bootstrap, t-SNE y validación cruzada |
| `--embeddings` | — | PCA y t-SNE de la penúltima capa |
| `--latencia-inception` | — | Mide el tiempo de Inception V3 por fotograma |
| `--decimal-punto` | — | Usa punto decimal en las tablas (por defecto coma, como en la tesis) |
| `--cv` | 0 | Número de pliegues de validación cruzada |
| `--cv-epochs`, `--cv-lr` | 60, 1e-4 | Épocas y tasa de aprendizaje en cada pliegue |
| `--cv-arch` | `original` | `original` (la de `train_lstm_harp.py`) o `ligera` |
| `--cv-por-persona` | — | Agrupa los pliegues por persona |
| `--cv-incluir-test` | — | Suma la partición test al conjunto de validación cruzada |
| `--cv-solo-originales` | — | Evalúa cada pliegue solo con videos originales |

La validación cruzada entrena desde cero en cada pliegue: con la arquitectura original y CPU puede tardar horas. Conviene probar primero con `--cv-epochs 5`.

## 6.3 Qué genera

```
metricas/20261006_221500/
├── resumen.md                       # informe legible con todas las cifras y figuras
├── resumen.json                     # todo en formato máquina
├── dataset_conteos.csv
├── cobertura_manos_por_video.csv    # fotogramas con mano, relleno y "fotogramas ajenos" (H2)
├── cobertura_manos_por_clase.csv
├── videos_crudos.csv                # fps, resolución, duración (con --raw-dir)
├── predicciones_test.csv            # video, real, predicha, confianza, top-3
├── metricas_por_clase_test.csv
├── tablas/
│   ├── tabla_dataset.tex
│   ├── tabla_cobertura_manos.tex
│   ├── tabla_metricas_globales_test.tex
│   ├── tabla_metricas_por_clase_test.tex
│   ├── tabla_matriz_confusion_test.tex
│   ├── tabla_resumen_particiones.tex
│   └── tabla_validacion_cruzada.tex          (con --cv)
└── figuras/  (PDF vectorial + PNG)
    ├── curvas_entrenamiento
    ├── cobertura_manos_por_clase
    ├── matriz_confusion_test / matriz_confusion_normalizada_test
    ├── metricas_por_clase_test
    ├── calibracion_test
    ├── roc_pr_test
    ├── embeddings_pca_tsne                    (con --embeddings)
    └── matriz_confusion_validacion_cruzada    (con --cv)
```

## 6.4 Qué significa cada métrica (y cómo leerla)

| Métrica | Qué responde | Lectura con este conjunto |
|---|---|---|
| **Exactitud** | ¿Qué proporción de videos acierta? | Con 20 videos, cada uno vale 5 puntos. Informar siempre “k de n” y el IC95 |
| **IC95 bootstrap** | ¿Entre qué valores estaría la exactitud real? | Remuestrea los videos de prueba 2000 veces. Si el intervalo incluye el 10 %, no se puede descartar que el modelo funcione como el azar |
| **Línea base 1/K** | ¿Cuánto acierta el azar? | 10 % con 10 señas |
| **p-valor binomial** | ¿Es la exactitud mayor que la del azar? | p < 0,05: diferencia significativa |
| **Exactitud balanceada** | Promedio de la exhaustividad de cada clase | Igual a la exactitud si las clases están equilibradas (como en test) |
| **Top-k** | ¿La seña correcta está entre las k más probables? | Útil para una aplicación que sugiera opciones |
| **Precisión** | Cuando el modelo dice “Rojo”, ¿cuántas veces tiene razón? | Por clase y promedio macro |
| **Exhaustividad** | De los “Rojo” reales, ¿cuántos encontró? | Con 2 videos por clase solo puede valer 0, 50 o 100 % |
| **F1** | Equilibrio entre precisión y exhaustividad | El promedio **macro** trata a todas las señas por igual |
| **Kappa de Cohen** | Acuerdo con la etiqueta real descontando el azar | 0 = azar, 1 = perfecto; > 0,6 se considera bueno |
| **MCC** | Correlación entre predicción y realidad | Entre −1 y 1; robusto con clases desequilibradas |
| **Log-loss** | ¿Qué tan seguras y acertadas son las probabilidades? | Menor es mejor; el azar con 10 clases da ln 10 ≈ 2,30 |
| **Brier** | Error cuadrático de las probabilidades | Menor es mejor; 0,9 equivale a repartir 1/10 a cada clase |
| **ROC-AUC (uno contra resto)** | ¿Ordena bien los videos de cada clase por probabilidad? | 0,5 = azar, 1 = perfecto |
| **ECE y diagrama de confiabilidad** | ¿Cuando dice 80 % de confianza, acierta el 80 %? | Importante si una aplicación muestra la confianza al usuario |
| **Matriz de confusión** | ¿Qué señas se confunden entre sí? | Analizar los pares más confundidos con criterios lingüísticos (configuración, movimiento, ubicación) |
| **Cobertura de manos** | ¿En cuántos fotogramas MediaPipe encontró manos? | Relacionar con los errores: Moryossef et al. (2021) observaron que los videos mal clasificados tenían menos fotogramas con mano detectada |
| **Latencia** | ¿Cuánto tarda en reconocer un video? | Separar MediaPipe, Inception (por fotograma × 150) y LSTM; relevante para un uso en tiempo real |
| **Parámetros y tamaño** | ¿Qué tan pesado es el modelo? | Comparar con los datos disponibles (sobreajuste) y con el estado del arte |
| **Silueta (embeddings)** | ¿La representación interna separa las clases? | Cerca de 1: grupos bien separados; cerca de 0: mezclados |
| **Validación cruzada** | ¿Qué tan estable es el resultado si cambian los videos de prueba? | Informar media ± desvío entre pliegues |

## 6.5 Cómo incluir los resultados en la tesis

En el preámbulo de `main.tex`:

```latex
\usepackage{booktabs}
\usepackage{graphicx}
```

Copiar la carpeta de resultados junto a la tesis (por ejemplo `resultados/`) y en el capítulo 6:

```latex
\input{resultados/tablas/tabla_metricas_globales_test.tex}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.75\textwidth]{resultados/figuras/matriz_confusion_normalizada_test.pdf}
  \caption{Matriz de confusión normalizada sobre el conjunto de prueba. Fuente: elaboración propia.}
  \label{fig:matriz-confusion-test}
\end{figure}
```

Las etiquetas de las tablas ya vienen definidas (`tab:metricas-globales-test`, `tab:metricas-clase-test`, `tab:matriz-confusion-test`, `tab:dataset-conteos`, `tab:cobertura-manos`, `tab:resumen-particiones`, `tab:validacion-cruzada`). Los números usan coma decimal y `\,\%`, igual que el resto de la tesis.

## 6.6 Estructura sugerida para el capítulo de resultados

1. **Conjunto de datos final:** `tabla_dataset.tex`, `tabla_cobertura_manos.tex`, figura de cobertura. Aclarar división antes del aumento y personas compartidas.
2. **Configuración experimental:** arquitectura (tabla de [01 · 1.5](01_arquitectura.md#15-arquitectura-del-clasificador-train_lstm_harppy)), hiperparámetros, hardware, versiones, semilla.
3. **Entrenamiento:** `curvas_entrenamiento.pdf`; comentar la brecha entre entrenamiento y validación (sobreajuste).
4. **Resultados en prueba:** `tabla_metricas_globales_test.tex` con IC95 y comparación con el azar; `tabla_metricas_por_clase_test.tex`; matriz de confusión.
5. **Robustez:** validación cruzada (`tabla_validacion_cruzada.tex`); si es posible, por persona.
6. **Análisis de errores:** pares más confundidos (`resumen.md`), `predicciones_test.csv`, relación con la cobertura de manos, calibración.
7. **Costo computacional:** parámetros, tamaño en disco, latencia por etapa.
8. **Comparación de variantes:** una tabla con una fila por experimento (versión anterior; `features2048` + `ligera`; `features2048` + `original`; `probs` (Inception V3 reentrenada); `landmarks` (coordenadas de MediaPipe); y, como línea base sin LSTM, la exactitud por video que imprime `retrain_inception.py`), con exactitud ± IC95 y F1 macro.
9. **Comparación con el estado del arte:** ubicar los resultados en la tabla 3.1, aclarando que la evaluación depende del signante.

## 6.7 Validación del script

El script se probó con un conjunto sintético que reproduce la estructura de `data/` (incluidos los nombres con `- Trim`, copias `_0`… y relleno `_NNNN`) y con los videos reales para el análisis de fps y resolución. La auditoría, las curvas, las métricas, las figuras y las tablas LaTeX (compiladas con `pdflatex`) funcionan. Las partes que dependen de TensorFlow real (carga de `lstm_senha_model`, embeddings, latencia de Inception y validación cruzada) se escribieron para la API de TF 2.15 del proyecto, pero no pudieron ejecutarse en el entorno de prueba: si alguna falla, el mensaje de error indicará la línea a revisar.

Actualización (6 de octubre de 2026): con el código corregido se ejecutó el pipeline completo (aumento, `handtrack.py`, extracción, `retrain_inception.py`, las tres representaciones, ambas arquitecturas, `predict_harp.py` y `evaluate_metrics.py` con `--cv`) sobre un subconjunto de 3 clases y 30 fotogramas por video, con TensorFlow 2.15 real. Todas las etapas funcionan; esa prueba no produce cifras útiles para la tesis.

## 6.8 Resultados de la regeneración completa (7 de octubre de 2026)

Datos: `rawdata/` (80 originales en train + 320 copias con rotación, zoom, eliminación de fotogramas y espejo; 20 originales en test), 150 fotogramas por video, semilla 42. Las métricas completas de cada modelo están en `metricas/<modelo>/` y los registros de cada etapa en `data/logs_pipeline/`.

### Prueba (20 videos, casi independiente del signante)

| Modelo (`--output`) | Representación | Arquitectura | Parámetros | Exactitud (aciertos) | IC95 | F1 macro | κ | p vs. azar |
|---|---|---|---:|---|---|---:|---:|---:|
| `lstm_senha_model` | `features2048` | ligera | 1 168 842 | 40 % (8/20) | 20 – 60 % | 0,29 | 0,33 | 0,0004 |
| `lstm_probs` | `probs` (Inception reentrenada) | ligera | 125 386 | 40 % (8/20) | 20 – 60 % | 0,34 | 0,33 | 0,0004 |
| `lstm_landmarks` | `landmarks` (MediaPipe) | ligera | 184 778 | 45 % (9/20) | 25 – 65 % | 0,41 | 0,39 | 0,00006 |
| `lstm_original` | `features2048` | original | 35 597 578 | 35 % (7/20) | 15 – 55 % | 0,21 | 0,28 | 0,002 |
| — (sin LSTM) | Inception V3 reentrenada, promedio por video | — | 20 490 | 35 % (7/20) | — | — | — | — |

Azar: 10 %. La versión anterior del código obtenía entre 18 % y 30 % (capítulo 6 de la tesis), con la prueba usada además como validación.

### Validación cruzada de 5 pliegues agrupada por persona

Los 100 originales (train + test) y sus copias; cada persona queda entera en un pliegue y se evalúa solo sobre originales. 60 épocas (`features2048`) o 100 (`landmarks`), arquitectura ligera, sin parada temprana.

| Representación | Exactitud media ± desvío | F1 macro | Pliegues |
|---|---|---|---|
| `features2048` | **53,5 % ± 6,2** | 0,48 ± 0,10 | 50,0 · 45,5 · 62,1 · 55,0 · 55,0 % |
| `landmarks` | 44,5 % ± 8,1 | 0,38 ± 0,08 | 35,0 · 54,5 · 37,9 · 50,0 · 45,0 % |

Comandos: `python evaluate_metrics.py --sin-modelo --data-type <tipo> --signers-csv personas_video.csv --cv 5 --cv-epochs <60|100> --cv-arch ligera --cv-por-persona --cv-incluir-test --cv-solo-originales` (salida en `metricas/cv_personas_<tipo>/`).

### Lectura

1. **Las correcciones mejoran el resultado** respecto de la versión anterior, y ahora con una evaluación más exigente: prueba con personas nuevas y validación separada.
2. **Con 20 videos de prueba no se pueden ordenar las variantes:** los intervalos se superponen por completo y la diferencia entre 35 %, 40 % y 45 % es de uno o dos videos. La validación cruzada por persona (100 videos) es la estimación más estable. Allí las características de Inception V3 superan a las coordenadas (53,5 % frente a 44,5 %), aunque el desvío entre pliegues es grande.
3. **La arquitectura liviana rinde igual o mejor que la original** con 30 veces menos parámetros (H6).
4. **Inception V3 reentrenada sola** (sin LSTM, promediando probabilidades por fotograma) acierta 35 % en prueba y 24 % en validación. La LSTM aporta al modelar el orden temporal. Las probabilidades reentrenadas (10 valores) rinden igual que las 2048 características en prueba, con un modelo 9 veces más chico.
5. Las confusiones más frecuentes son entre señas parecidas en el movimiento de la mano (agosto, rojo y sábado → marzo; broma → jugar; miércoles → soltero). Ver `metricas/<modelo>/resumen.md`.
6. En todos los modelos la exactitud sobre train (60 % – 73 %) supera ampliamente a la de prueba: sigue habiendo sobreajuste, esperable con 80 grabaciones de entrenamiento.

## 6.9 Estabilidad entre semillas y señas parecidas (7 de octubre de 2026)

Sugerencia del tutor: repetir el entrenamiento con varias semillas e informar media ± desvío estándar. Se usaron las semillas 42, 43, 44, 45 y 46.

**Cómo reproducirlo:**

```bash
./experimentos_semillas.sh        # ~9 h sin GPU; se puede interrumpir y relanzar (salta lo ya hecho)
python analisis_semillas.py       # tablas, figuras y resumen en metricas/analisis_semillas/
```

- En los modelos evaluados sobre la prueba, la semilla cambia la inicialización de los pesos y el orden de los lotes. La separación train/validación es la misma en todas (`--split-seed 42`), así que la variación se debe solo al entrenamiento.
- En la validación cruzada por persona, la semilla cambia además qué personas caen en cada pliegue (validación cruzada repetida).
- La capa reentrenada de Inception V3 (`probs`) es la misma en todas las semillas (semilla 42).
- Cada caso tiene sus métricas completas en `metricas/semillas/<experimento>/s<semilla>/` y `metricas/semillas_cv/<experimento>/s<semilla>/`.

### Resultados: 10 señas

| Modelo | Prueba (20 videos) | Aciertos (mín. – máx.) | Validación cruzada por persona (100 videos) |
|---|---|---|---|
| Inception V3 (2048), LSTM ligera | 44,0 % ± 9,6 | 7 – 12 | **53,0 % ± 1,6** |
| Inception V3 (2048), LSTM original | 40,0 % ± 5,0 | 7 – 9 | — |
| Coordenadas de MediaPipe, LSTM ligera | 47,0 % ± 7,6 | 7 – 11 | 45,6 % ± 4,6 |
| Inception V3 reentrenada (probabilidades), LSTM ligera | 38,0 % ± 5,7 | 6 – 9 | — |

Media ± desvío estándar entre las 5 semillas; azar: 10 %. Tablas: `tabla_semillas_test.tex`, `tabla_semillas_cv.tex`; figuras: `semillas_exactitud_test.pdf`, `semillas_exactitud_cv.pdf`.

**Lectura:**

1. **La semilla sola mueve la exactitud en prueba hasta 25 puntos:** el modelo principal va de 35 % a 60 % según la semilla. Un único entrenamiento con semilla 42 (40 %) no basta para caracterizar el modelo, como señaló el tutor.
2. En la prueba, las diferencias entre representaciones (38 % – 47 %) son menores que el desvío de cada una: **con 20 videos no se pueden ordenar**.
3. **La validación cruzada por persona es mucho más estable** (desvío de 1,6 puntos entre semillas para Inception V3) y sí separa las representaciones: las características de Inception V3 (53,0 %) superan a las coordenadas de MediaPipe (45,6 %). Es la cifra recomendada para el resultado principal, junto con la de prueba como evaluación independiente.
4. Dentro de cada validación cruzada, el desvío entre pliegues (≈ 10 – 12 puntos) es mayor que entre semillas: el rendimiento depende bastante de **qué personas** quedan fuera.
5. La arquitectura original (35,6 M parámetros) no supera a la ligera (1,2 M).
6. Las probabilidades de la capa reentrenada (10 valores por fotograma) rinden algo menos que las 2048 características: comprimir a 10 números pierde información útil para la LSTM.

### Señas parecidas

Pares indicados como parecidos: (marzo, rojo), (sábado, agosto), (jugar, broma), (nombre, martes), (miércoles, soltero). Tablas: `tabla_parecidas_test.tex`, `tabla_parecidas_cv.tex`, `tabla_no_parecidas_*.tex`; matrices de confusión ordenadas por par: `confusion_pares_*.pdf`.

Validación cruzada por persona, Inception V3 (2048), media de 5 semillas:

| Medida | Resultado | Azar |
|---|---|---|
| Exactitud con 10 señas | 52,4 % ± 3,1 | 10 % |
| Exactitud a nivel de par (se acepta confundir una seña con su pareja) | 79,2 % ± 4,4 | 20 % |
| Errores que son confusiones dentro del par | 56,6 % ± 7,1 | — |
| Distinguir las dos señas de un par | 64,1 % ± 2,7 | 50 % |
| Señas no parecidas: 5 señas, una de cada par, modelo de 10 señas (32 combinaciones) | 77,8 % ± 2,9 | 20 % |
| Señas no parecidas: modelo **entrenado** solo con el grupo A (marzo, sábado, jugar, nombre, miércoles) | 79,2 % ± 5,4 | 20 % |
| Señas no parecidas: modelo **entrenado** solo con el grupo B (rojo, agosto, broma, martes, soltero) | 76,6 % ± 2,8 | 20 % |

En la prueba (20 videos) el patrón es el mismo: 44 % con 10 señas, 78 % a nivel de par, 74 % – 80 % con los modelos entrenados solo con señas no parecidas.

La exactitud con 10 señas de esta tabla (52,4 %) se calcula sobre todas las predicciones juntas; la de la tabla anterior (53,0 %) es el promedio de los pliegues, que tienen tamaños distintos.

Capacidad de distinguir las dos señas de cada par (validación cruzada, media de 5 semillas):

| Par | Inception V3 (2048) | Coordenadas de MediaPipe |
|---|---:|---:|
| marzo – rojo | 57 % | 52 % |
| sábado – agosto | 60 % | 66 % |
| jugar – broma | 63 % | 56 % |
| nombre – martes | **87 %** | **97 %** |
| miércoles – soltero | 53 % | 57 % |

**Lectura:**

1. **La mayor parte del error viene de las señas parecidas.** Cuando las señas no se parecen, el sistema acierta alrededor del 78 % con 5 clases, el triple de lo esperable por azar. Con las 10 señas baja a ≈ 53 % porque más de la mitad de los errores son confusiones con la pareja.
2. **Entrenar solo con señas no parecidas no mejora respecto de usar el modelo de 10 señas restringido a esas 5** (grupo A: 79 % entrenado solo con esas señas frente a 80 % con el modelo de 10 señas restringido; grupo B: 77 % en ambos casos). La dificultad está en la similitud de los pares, no en la cantidad de clases.
3. **No todos los pares son igual de difíciles:**
   - **miércoles – soltero, marzo – rojo y jugar – broma** son prácticamente indistinguibles (53 % – 63 %, cerca del azar de 50 %).
   - **nombre – martes** se distingue bien (87 % – 97 %): para el sistema no es un par confuso.
4. **marzo, rojo, sábado y agosto forman un grupo de cuatro señas que se confunden entre sí**, no dos pares separados. Sábado y agosto se confunden entre sí (24 % y 18 %), pero también con marzo: agosto se predice como marzo el 30 % de las veces y sábado el 24 % (ver `confusion_pares_cv_cv_features2048.pdf`). Por eso la exactitud “a nivel de par” subestima la dificultad de este grupo.
5. Las coordenadas de MediaPipe separan mejor nombre – martes y sábado – agosto, y peor marzo – rojo y jugar – broma. Sugiere que esas diferencias dependen de la forma o la orientación de la mano, que cada representación capta de manera distinta.

**Texto sugerido para la tesis:**

> Para evaluar la estabilidad del entrenamiento, cada configuración se entrenó con cinco semillas distintas (42 a 46) y se informa la media y el desvío estándar. En el conjunto de prueba, el modelo con características de Inception V3 alcanzó una exactitud de 44,0 % ± 9,6 % (entre 35 % y 60 % según la semilla), lo que muestra que, con 20 videos de prueba, un único entrenamiento no es representativo. La validación cruzada de cinco pliegues agrupada por persona, repetida con las mismas cinco semillas, resultó mucho más estable: 53,0 % ± 1,6 %.
>
> El análisis de errores muestra que la mayor parte de ellos se concentra en pares de señas de movimiento similar. Si se acepta como correcta la confusión entre las señas de un mismo par, la exactitud asciende a 79,2 % ± 4,4 %; entre cinco señas no parecidas entre sí, el sistema alcanza 77,8 % ± 2,9 %, frente a un 20 % esperable por azar. Los pares miércoles–soltero, marzo–rojo y jugar–broma resultaron prácticamente indistinguibles para el modelo, mientras que nombre–martes se distinguió en el 87 % de los casos.
