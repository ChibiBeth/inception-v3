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

## 6.8 Resultados con el conjunto equilibrado (8 de octubre de 2026)

Datos: `rawdata/` equilibrado (10 personas × 10 señas; 8 personas en train y 2 en test). En train hay 80 originales y 320 copias aumentadas (rotación, zoom, eliminación de fotogramas y espejo). Se usan 150 fotogramas por video. Todo se generó desde cero con `./pipeline_completo.sh`.

- Métricas completas de cada modelo: `metricas/<modelo>/`.
- Registros de cada etapa: `data/logs_pipeline/`.
- Resultados del conjunto del 7 de octubre (con `marzo_cr1_2` y sin `martes_cr5`): respaldados en `../respaldo_dataset_v2_2026-10-07/`.

### Prueba, semilla 42 (20 videos de 2 personas que no aparecen en train)

| Modelo (`--output`) | Representación | Arquitectura | Parámetros | Exactitud (aciertos) | IC95 | F1 macro | κ | p vs. azar |
|---|---|---|---:|---|---|---:|---:|---:|
| `lstm_senha_model` | `features2048` | ligera | 1 168 842 | 35 % (7/20) | 15 – 55 % | 0,30 | 0,28 | 0,002 |
| `lstm_probs` | `probs` (Inception reentrenada) | ligera | 125 386 | 35 % (7/20) | 15 – 55 % | 0,29 | 0,28 | 0,002 |
| `lstm_landmarks` | `landmarks` (MediaPipe) | ligera | 184 778 | 50 % (10/20) | 30 – 70 % | 0,40 | 0,44 | 0,000007 |
| `lstm_original` | `features2048` | original | 35 597 578 | 45 % (9/20) | 25 – 70 % | 0,39 | 0,39 | 0,00006 |
| — (sin LSTM) | Inception V3 reentrenada, promedio por video | — | 20 490 | 40 % (8/20) | — | — | — | — |

Azar: 10 %. Con una sola semilla y 20 videos estas cifras son muy inestables: ver 6.9 para la media entre 5 semillas, que es lo que conviene informar.

## 6.9 Estabilidad entre semillas y señas parecidas (8 de octubre de 2026)

Sugerencia del tutor: repetir el entrenamiento con varias semillas e informar media ± desvío estándar. Se usaron las semillas 42, 43, 44, 45 y 46.

**Cómo reproducirlo:**

```bash
./pipeline_completo.sh            # desde los videos; incluye experimentos_semillas.sh y analisis_semillas.py (~12 h sin GPU)
./experimentos_semillas.sh        # solo la batería de semillas (requiere los datos ya generados)
python analisis_semillas.py       # tablas, figuras y resumen en metricas/analisis_semillas/
```

- En los modelos evaluados sobre la prueba, la semilla cambia la inicialización de los pesos y el orden de los lotes. La separación train/validación es la misma en todas (`--split-seed 42`).
- En la validación cruzada por persona (10 personas, 5 pliegues de 2 personas, todos de 20 videos), la semilla cambia además qué personas caen en cada pliegue.
- La capa reentrenada de Inception V3 (`probs`) es la misma en todas las semillas (semilla 42).
- Cada caso tiene sus métricas completas en `metricas/semillas/<experimento>/s<semilla>/` y `metricas/semillas_cv/<experimento>/s<semilla>/`.
- Resumen completo en `metricas/analisis_semillas/resumen_semillas.md`.

### Resultados: 10 señas

| Modelo | Prueba (20 videos) | Aciertos (mín. – máx.) | Validación cruzada por persona (100 videos) |
|---|---|---|---|
| Inception V3 (2048), LSTM ligera | 42,0 % ± 10,4 | 6 – 11 | **54,0 % ± 2,8** |
| Inception V3 (2048), LSTM original | 43,0 % ± 7,6 | 7 – 11 | — |
| Coordenadas de MediaPipe, LSTM ligera | **52,0 % ± 2,7** | 10 – 11 | 49,2 % ± 2,5 |
| Inception V3 reentrenada (probabilidades), LSTM ligera | 39,0 % ± 6,5 | 6 – 9 | — |

Media ± desvío estándar entre las 5 semillas; azar: 10 %. Las dos evaluaciones son independientes del signante. Tablas: `tabla_semillas_test.tex`, `tabla_semillas_cv.tex`; figuras: `semillas_exactitud_test.pdf`, `semillas_exactitud_cv.pdf`.

**Lectura:**

1. **La semilla sola mueve la exactitud en prueba hasta 25 puntos.** Con Inception V3 (2048) y LSTM ligera va de 30 % a 55 % según la semilla, así que un único entrenamiento no caracteriza el modelo, como señaló el tutor.
2. **En la prueba**, las coordenadas de MediaPipe son las más estables (52 % ± 2,7) y las de mejor media. Aun así, sobre 20 videos de solo 2 personas la diferencia con Inception (42 % ± 10,4) es de dos aciertos en promedio.
3. **En la validación cruzada por persona** (10 personas, 100 videos), el orden se invierte: Inception V3 (54,0 % ± 2,8) supera a las coordenadas (49,2 % ± 2,5). Es la estimación más confiable, porque promedia sobre las 10 personas en lugar de 2. Que el orden cambie entre ambas evaluaciones indica que **la diferencia entre representaciones es pequeña y depende de qué personas se evalúan**. En la tesis conviene presentarlas como de rendimiento comparable, con una ligera ventaja de Inception V3 en la evaluación más amplia.
4. Dentro de cada validación cruzada, el desvío entre pliegues (≈ 8 puntos) es mayor que entre semillas: el rendimiento depende de qué personas quedan fuera.
5. La arquitectura original (35,6 M parámetros) rinde igual que la ligera (1,2 M) en media (43 % frente a 42 %), con 30 veces más parámetros y 15 veces más tiempo de entrenamiento.
6. Las probabilidades de la capa reentrenada (10 valores por fotograma) rinden algo menos que las 2048 características: comprimir a 10 números pierde información útil para la LSTM.
7. En train la exactitud llega a 54 % – 83 % según el modelo, muy por encima de la prueba: hay sobreajuste, esperable con 8 personas de entrenamiento.

### Señas parecidas

Pares indicados como parecidos: (marzo, rojo), (sábado, agosto), (jugar, broma), (nombre, martes), (miércoles, soltero). Tablas: `tabla_parecidas_test.tex`, `tabla_parecidas_cv.tex`, `tabla_no_parecidas_*.tex`; matrices de confusión ordenadas por par: `confusion_pares_*.pdf`.

Validación cruzada por persona, media de 5 semillas:

| Medida | Inception V3 (2048) | Coordenadas de MediaPipe | Azar |
|---|---|---|---|
| Exactitud con 10 señas | 54,0 % ± 2,8 | 49,2 % ± 2,5 | 10 % |
| Exactitud a nivel de par (se acepta confundir una seña con su pareja) | 80,2 % ± 2,9 | 65,2 % ± 3,4 | 20 % |
| Errores que son confusiones dentro del par | 57,1 % ± 4,4 | 31,6 % ± 4,3 | — |
| Distinguir las dos señas de un par | 64,8 % ± 2,5 | 69,4 % ± 1,9 | 50 % |
| Señas no parecidas: 5 señas, una de cada par, modelo de 10 señas (32 combinaciones) | 79,3 % ± 2,3 | 66,1 % ± 3,4 | 20 % |
| Señas no parecidas: modelo **entrenado** solo con el grupo A (marzo, sábado, jugar, nombre, miércoles) | 78,8 % ± 4,8 | — | 20 % |
| Señas no parecidas: modelo **entrenado** solo con el grupo B (rojo, agosto, broma, martes, soltero) | 79,6 % ± 4,3 | — | 20 % |

En la prueba (20 videos), con Inception V3 (2048) y LSTM ligera, el patrón es el mismo:
- 42 % con 10 señas y 81 % a nivel de par.
- Dos tercios de los errores, dentro del par.
- 76 % entre señas no parecidas.
- Los modelos entrenados solo con señas no parecidas dan 64 % ± 11 en el grupo A y 90 % ± 0 en el B, con solo 10 videos por grupo.

Capacidad de distinguir las dos señas de cada par (validación cruzada, media de 5 semillas):

| Par | Inception V3 (2048) | Coordenadas de MediaPipe |
|---|---:|---:|
| marzo – rojo | 53 % | 54 % |
| sábado – agosto | 67 % | 70 % |
| jugar – broma | 56 % | 58 % |
| nombre – martes | **92 %** | **98 %** |
| miércoles – soltero | 56 % | 67 % |

**Lectura:**

1. **Con Inception V3, la mayor parte del error viene de las señas parecidas.** Entre señas no parecidas el sistema acierta ≈ 79 % con 5 clases, cuatro veces lo esperable por azar. Con las 10 señas baja a 54 % porque el 57 % de los errores son confusiones con la pareja. Si se acepta como correcta la confusión dentro del par, la exactitud sube a 80 %.
2. **Entrenar solo con señas no parecidas no mejora respecto de usar el modelo de 10 señas restringido a esas 5.** En el grupo A da 79 % entrenando solo con esas señas y 81 % con el modelo restringido; en el grupo B, 80 % y 78 %. La dificultad está en la similitud de los pares, no en la cantidad de clases.
3. **No todos los pares son igual de difíciles:**
   - **marzo – rojo, jugar – broma y miércoles – soltero** son casi indistinguibles (53 % – 67 %, frente a un azar de 50 %).
   - **sábado – agosto** se distingue algo mejor (67 % – 70 %).
   - **nombre – martes** se distingue muy bien (92 % – 98 %): para el sistema no es un par confuso.
4. **marzo, rojo, sábado y agosto forman un grupo de cuatro señas que se confunden entre sí**, no dos pares separados. Con Inception V3, marzo se predice como sábado el 24 % de las veces, sábado como marzo el 28 % y agosto como marzo el 22 % (`confusion_pares_cv_cv_features2048.pdf`). Por eso la exactitud “a nivel de par” subestima la dificultad de este grupo.
5. **Las coordenadas de MediaPipe se equivocan de otra manera.** Solo el 32 % de sus errores cae dentro del par, frente al 57 % de Inception V3, y distinguen mejor dentro de cada par (69 %). En cambio, confunden más señas no emparejadas: marzo casi nunca se reconoce (8 %) y se reparte entre rojo, sábado y nombre; nombre atrae errores de marzo, rojo y agosto (`confusion_pares_cv_cv_landmarks.pdf`). Las dos representaciones fallan en señas distintas, lo que sugiere que combinarlas podría mejorar el resultado. Es un trabajo futuro posible.

**Texto sugerido para la tesis:**

> Para evaluar la estabilidad del entrenamiento, cada configuración se entrenó con cinco semillas distintas (42 a 46) y se informa la media y el desvío estándar. En el conjunto de prueba, formado por dos personas que no participaron del entrenamiento, el modelo con características de Inception V3 obtuvo una exactitud de 42,0 % ± 10,4 % (entre 30 % y 55 % según la semilla), lo que muestra que, con 20 videos de prueba, un único entrenamiento no es representativo. La validación cruzada de cinco pliegues agrupada por persona, que evalúa a cada una de las diez personas sin haberla visto durante el entrenamiento, resultó mucho más estable: 54,0 % ± 2,8 % con características de Inception V3 y 49,2 % ± 2,5 % con las coordenadas de MediaPipe.
>
> El análisis de errores muestra que la mayor parte de ellos se concentra en pares de señas de movimiento similar. Con Inception V3, si se acepta como correcta la confusión entre las señas de un mismo par, la exactitud asciende a 80,2 % ± 2,9 %. Entre cinco señas no parecidas entre sí, el sistema alcanza 79,3 % ± 2,3 %, frente a un 20 % esperable por azar, y entrenar un modelo solo con esas señas no mejora este resultado. Los pares marzo–rojo, jugar–broma y miércoles–soltero resultaron prácticamente indistinguibles, mientras que nombre–martes se distinguió en más del 90 % de los casos.
