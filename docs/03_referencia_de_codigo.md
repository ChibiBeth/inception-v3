# 3. Referencia del código

Cada sección describe un archivo: para qué sirve, qué recibe, qué produce y cómo funciona por dentro. Los códigos H1, H2… remiten a [05 · Hallazgos y recomendaciones](05_hallazgos_y_recomendaciones.md), donde se explica qué problema tenía la versión anterior y cómo se corrigió.

---

## 3.1 `lspy_common.py`

**Propósito:** utilidades compartidas, sin dependencias pesadas.

| Elemento | Qué hace |
|---|---|
| `DATA_TYPES` | Representaciones que puede usar la LSTM: `features2048`, `probs`, `landmarks` y `features` (versión anterior) |
| `set_seed(seed)` | Fija las semillas de Python, NumPy y TensorFlow (H13) |
| `original_stem(nombre)` / `is_augmented(nombre)` | `agosto_cr1_0` → `agosto_cr1`; `marzo_cr4-Trim_2` → `marzo_cr4`. Un nombre con ` - Trim` (con espacios) siempre es original |
| `video_original(partición, nombre)` / `is_augmented_row(partición, nombre)` | Igual que las anteriores, pero solo train tiene copias: en test `marzo_cr1_2` es un original |
| `sequence_file(data_dir, partición, nombre, T, tipo)` | `data/sequences/<partición>/<nombre>-<T>-<tipo>.npy`. La subcarpeta por partición evita que la copia `marzo_cr1_2` de train pise al original `marzo_cr1_2` de test |
| `frame_pattern(nombre)` | Patrón `glob` de los fotogramas de **un** video: `<nombre>[-_]NNNN.jpg` (H2) |
| `grouped_train_val_split(nombres, clases, val_frac, seed)` | Elige al azar (con semilla) el `val_frac` de los originales de cada clase (al menos uno) y devuelve los índices de train y validación; las copias acompañan a su original (H3) |
| `save_meta(modelo, dict)` / `load_meta(modelo)` | Guardan y leen `<modelo>.json` con clases y configuración (H10) |

---

## 3.2 `data_augmentation.py`

**Propósito:** generar variaciones de cada video original de entrenamiento con las técnicas que describe la tesis: rotación, zoom y eliminación aleatoria de fotogramas.

**Uso:** `python data_augmentation.py --main-folder-path <entrada> --output-folder-path <salida> --max-clips N [opciones]`

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--main-folder-path` | (obligatorio) | Carpeta de una clase, o partición con una subcarpeta por clase |
| `--output-folder-path` | (obligatorio) | Carpeta destino; replica las subcarpetas de clase. **No se borra** |
| `--max-clips` | (obligatorio) | Copias por video original |
| `--max-rotacion` | 30 | Ángulo máximo, en grados |
| `--max-zoom` | 0.15 | Zoom entre 1 − z y 1 + z |
| `--max-eliminacion` | 0.2 | Fracción máxima de fotogramas eliminados |
| `--espejo` | no | Espejar horizontalmente las copias de índice par |
| `--copiar-originales` | no | Copiar también los originales a la salida |
| `--incluir-aumentados` | no | Aumentar también videos que ya son copias |
| `--sobrescribir` | no | Reemplazar copias existentes |
| `--semilla`, `--hilos` | 42, 4 | |

**Funciones**

| Función | Qué hace |
|---|---|
| `augmentation_params(nombre, i, opt)` | Sortea ángulo, zoom y fracción eliminada de la copia `i` con un generador inicializado con `(semilla, nombre, i)`: el resultado no depende del orden en que terminan los hilos |
| `augment_and_save_frames(ruta, salida, params)` | Lee el video cuadro a cuadro; descarta cada cuadro con probabilidad igual a la fracción sorteada; aplica espejo (si corresponde) y una única transformación afín (rotación + zoom alrededor del centro, bordes negros); escribe un `.mp4` (`mp4v`) con los mismos fps y tamaño |
| `augment_video(ruta, carpeta, i, opt)` | Arma el nombre `<nombre sin espacios>_<i>.mp4`, salta si existe y devuelve la fila para el registro |
| `find_videos(carpeta, incluir_aumentados)` | Lista los videos (de una clase o de todas) y omite las copias |
| bloque `__main__` | Valida que la salida sea distinta de la entrada, copia originales si se pidió, ejecuta las copias en paralelo (`ThreadPoolExecutor`) y agrega las filas a `<salida>/aumento_parametros.csv` |

**Registro `aumento_parametros.csv`:** `video_original, copia, espejo, angulo, zoom, fraccion_eliminada, fotogramas_original, fotogramas_copia`.

---

## 3.3 `handtrack.py`

**Propósito:** convertir cada video en una secuencia de dibujos de las manos (y de coordenadas) de longitud fija.

**Uso:** `python handtrack.py -i <carpeta_videos> -o <carpeta_salida> [-n 150]`

**Funciones**

| Función | Qué hace |
|---|---|
| `get_video_parts(ruta, vid_dir)` | Con `os.path.relpath` y `os.path.splitext` obtiene `(partición, clase, nombre_sin_extensión, nombre)`; funciona con cualquier ruta (H12) |
| `detect_hands(cuadro, hands)` | Espeja el cuadro, lo completa con bordes negros hasta hacerlo cuadrado y ejecuta MediaPipe sobre la imagen en su resolución original (H8) |
| `draw_hands(results)` | Lienzo negro de 299 × 299 con los 21 puntos y sus conexiones en blanco (grosor 1, radio 1) |
| `landmarks_vector(results)` | 126 valores: mano izquierda y derecha (según `multi_handedness`) × 21 puntos × (x, y, z); muñeca en coordenadas del cuadro y los demás puntos relativos a la muñeca; ceros si falta una mano (H9) |
| `process_video(ruta, n)` | Ejecuta MediaPipe en **todos** los cuadros (el seguimiento necesita cuadros consecutivos), elige `n` cuadros equiespaciados con `np.linspace` y devuelve dibujos, coordenadas y datos del video (H7). Lo reutiliza `predict_harp.py` |
| `hands_extraction(vid_dir, out_dir, n)` | Recorre `train/` y `test/`, borra los fotogramas y `.npy` anteriores de cada video, guarda `<video>-NNNN.jpg` (NNNN = 0001 … n), `sequences/<partición>/<video>-<n>-landmarks.npy` y `data_file.csv` |
| `main()` | Argumentos `-i`, `-o`, `-n` con `argparse` |

**`data_file.csv`:** `partición, clase, video, n, fotogramas con mano (de los n), fotogramas del video, fps`. Las tres últimas columnas son informativas; `DataSet` solo usa las cuatro primeras.

**Detalle:** el espejo horizontal (`cv2.flip`) se aplica a todos los videos, en el entrenamiento y en la predicción; no cambia lo que aprende el modelo, pero hace que la etiqueta “Left/Right” de MediaPipe corresponda a la mano real de la persona.

---

## 3.4 `extract_features_harp.py`

**Propósito:** convertir cada secuencia de imágenes en una matriz de vectores de Inception V3. También define la clase `DataSet` que usan los demás scripts.

**Clase `DataSet(seq_length=150, class_limit=None, image_shape=(299, 299, 3), max_frames=None)`**

| Método | Qué hace |
|---|---|
| `get_data()` | Lee `data/data_file.csv` |
| `get_classes()` | Lista ordenada alfabéticamente de clases; si `class_limit` está definido, toma las primeras |
| `clean_data()` | Se queda con las filas con al menos `seq_length` fotogramas (y como máximo `max_frames`, si se indica) y cuya clase está en la lista |
| `get_class_one_hot(clase)` | Índice de la clase → vector *one-hot* |
| `split_train_test()` | Separa por la primera columna (`train` / cualquier otra cosa = test) |
| `get_all_sequences_in_memory(split, data_type)` | Carga todos los `.npy` de una partición → `(X, y)` |
| `load_rows(filas, data_type)` | Carga los `.npy` de las filas indicadas → `(X, y)`; error claro si falta alguno |
| `sequence_file(fila, data_type)` | `data/sequences/<partición>/<nombre>-<seq_length>-<data_type>.npy` |
| `get_frames_for_sample(fila)` | Fotogramas del video con `frame_pattern` (H2) |
| `rescale_list(lista, size)` | `size` elementos equiespaciados de toda la lista |

**Funciones**

| Función | Qué hace |
|---|---|
| `build_extractor()` | `InceptionV3(weights='imagenet', include_top=False, pooling='avg')`: 2048 valores por imagen, sin capas aleatorias (H1) |
| `load_frames(rutas)` | Lee los `.jpg` como RGB |
| `canvases_to_rgb(dibujos)` | Para dibujos en memoria (predicción): los pasa por JPEG y a RGB, igual que en el entrenamiento (H10) |
| `extract_features(modelo, imágenes)` | `preprocess_input` y `predict` en lotes de 32 (H14) |
| `main()` | Para cada video sin `.npy`, extrae y guarda `<video>-<seq_length>-features2048.npy`. Argumentos: `--seq-length`, `--class-limit`, `--batch-size` |

---

## 3.5 `retrain_inception.py`

**Propósito:** reentrenar la última capa de Inception V3 para que, además de las características, entregue la probabilidad de cada seña por fotograma (rol descrito en la tesis).

**Uso:** `python retrain_inception.py [--seq-length 150] [--folds 5] [--epochs 50] [--lr 1e-3]`

**Flujo:**

1. Carga las secuencias `features2048` de todos los videos.
2. Calcula el vector de Inception de un dibujo negro y marca como “sin mano” los fotogramas iguales a él; esos fotogramas no se usan para entrenar.
3. Separa train / validación con `grouped_train_val_split` (la misma separación que `train_lstm_harp.py` si se usan los mismos `--val-frac` y `--seed`).
4. Entrena `Dropout(0.5) → Dense(10, softmax)` (regularización L2) con los fotogramas de train; la validación decide cuándo parar. Guarda `data/inception_head.keras` y su `.json`.
5. Calcula las probabilidades de todos los videos; las de los videos de train se reemplazan por predicciones **fuera de pliegue** (`--folds` capas, cada una entrenada sin un grupo de originales).
6. Guarda `<video>-<seq_length>-probs.npy` e imprime la exactitud de Inception V3 sola en validación y prueba (por fotograma y por video promediando probabilidades).

---

## 3.6 `train_lstm_harp.py`

**Propósito:** entrenar la red recurrente con las secuencias.

**Flujo:**

1. Lee los argumentos (ver [02 · Paso 5](02_instalacion_y_uso.md#paso-5--entrenamiento-de-la-lstm)) y fija la semilla.
2. Define *callbacks*: `ModelCheckpoint` (mejor `val_loss`), `TensorBoard`, `EarlyStopping(restore_best_weights=True)` y `CSVLogger`, todos con el nombre `lstm-<data_type>-<arch>`.
3. Separa una validación desde train con `grouped_train_val_split` (H3) y carga train, validación y prueba.
4. Construye la arquitectura `ligera` u `original` con `Input(shape=X.shape[1:])`, de modo que acepta cualquier representación (H6).
5. Entrena con Adam (`--lr`) y valida con la validación.
6. Evalúa **una vez** sobre la prueba, guarda el modelo (con los mejores pesos) y `<modelo>.json`.

---

## 3.7 `predict_harp.py`

**Propósito:** predecir la seña de un video suelto.

**Uso:** `python predict_harp.py <video> [--model lstm_senha_model] [--top 3]`

**Flujo:**

1. Carga el modelo y su `.json` (clases, representación y longitud). Si no hay `.json`, usa las carpetas de `data/train` y `features2048`.
2. `process_video` de `handtrack.py`: mismos pasos que en el entrenamiento.
3. Según la representación: coordenadas directas, o Inception V3 (`build_extractor`, tras pasar los dibujos por JPEG) y, para `probs`, la capa de `data/inception_head.keras`.
4. Imprime las `--top` señas más probables y la ganadora.

---

## 3.8 `experimentos_semillas.sh` y `analisis_semillas.py`

`experimentos_semillas.sh` entrena y evalúa cada variante con las semillas de `SEMILLAS` (42 a 46 por defecto) y la validación cruzada por persona. También entrena modelos de 5 clases solo con señas no parecidas (grupo A: una seña de cada par; grupo B: la otra). Reutiliza los modelos de semilla 42 del pipeline principal y salta lo que ya existe.

`analisis_semillas.py` lee `metricas/semillas*/` y escribe en `metricas/analisis_semillas/`:

| Función | Qué hace |
|---|---|
| `seed_dirs(base)` | Encuentra `<experimento>/s<semilla>/` con `resumen.json` |
| `read_probs(path)` | Lee `probabilidades_test.csv` o `probabilidades_validacion_cruzada.csv` (vector completo de probabilidades por video, generado por `evaluate_metrics.py`) |
| `restricted_accuracy(...)` | Exactitud sobre los videos de un subconjunto de señas, eligiendo solo entre esas señas |
| `pair_analysis(...)` | Exactitud con 10 señas y a nivel de par, porcentaje de errores dentro del par, distinción dentro de cada par y exactitud en los 32 subconjuntos de señas no parecidas |
| `plot_pair_confusion`, `plot_seeds` | Matriz de confusión ordenada por pares (suma de semillas) y exactitud por semilla |

---

## 3.9 `loadpicklefileanddisplay.py`

Muestra la longitud y el primer elemento de un `.pickle`, y opcionalmente todos. No forma parte del pipeline actual.

---

## 3.10 `evaluate_metrics.py`

Ver [06 · Métricas para la tesis](06_metricas_para_la_tesis.md). Resumen de su estructura:

| Bloque | Funciones principales |
|---|---|
| Utilidades | `latex_table`, `fmt_pct`, `source_tag`, `load_signers_csv` (y `original_stem`, `is_augmented`, `frame_pattern` de `lspy_common`) |
| Datos | `read_data_file`, `filter_rows` (misma lógica que `DataSet`), `load_sequences` |
| 1. Auditoría | `audit_dataset`: conteos, fuga, grupos, cobertura de manos (usa la columna 5 de `data_file.csv` si existe), colisiones de nombre, estadísticas de características, videos crudos |
| 2. Entrenamiento | `training_curves`: lee el `.log` de CSVLogger del mismo tipo de modelo y grafica pérdida y exactitud |
| 3. Evaluación | `load_model_any`, `predict_proba`, `compute_metrics`, `evaluate_split`, `measure_latency`, `inception_latency`, figuras (`plot_confusion`, `plot_per_class`, `plot_reliability`, `plot_roc_pr`) y tablas (`write_metric_tables`) |
| 4. Embeddings | `plot_embeddings`: PCA y t-SNE de la penúltima capa + coeficiente de silueta |
| 5. Validación cruzada | `build_model` (arquitectura original o ligera), `cross_validation` con `StratifiedGroupKFold` |
| Informe | `write_markdown`, `resumen.json` |

`--data-type` y `--clases` se toman por defecto del `.json` del modelo (o `features2048` y todas las señas). Además de las tablas, guarda `probabilidades_<partición>.csv` y, en la validación cruzada, `probabilidades_validacion_cruzada.csv` con el vector completo de probabilidades de cada video.
