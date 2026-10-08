# 5. Hallazgos y recomendaciones

Revisión completa del código y de los datos (6 de octubre de 2026). Cada hallazgo indica su gravedad, dónde está, por qué importa y cómo corregirlo. Al final hay una hoja de ruta priorizada y una lista de puntos de la tesis que conviene alinear con el código.

| Código | Gravedad | Resumen |
|---|---|---|
| [H1](#h1) | Crítica | Las “características” salen de una capa `Dense(10)` con pesos aleatorios, distinta en cada ejecución |
| [H2](#h2) | Crítica | El `glob` de `extract_features_harp.py` mezcla fotogramas de varios videos |
| [H3](#h3) | Alta | El conjunto de prueba se usa como validación (elige época y detiene el entrenamiento) |
| [H4](#h4) | Alta | Solo 20 videos de prueba: cada error cambia la exactitud en 5 puntos |
| [H5](#h5) | Alta | Evaluación dependiente del signante (mismas personas en train y test) |
| [H6](#h6) | Media | Modelo sobredimensionado (18,9 M parámetros para ≈ 400 secuencias) y tasa de aprendizaje muy baja |
| [H7](#h7) | Media | Se descartan los fotogramas sin manos y se rellena; fps distintos cambian la escala temporal |
| [H8](#h8) | Media | Los videos verticales se deforman a 299 × 299 antes de MediaPipe |
| [H9](#h9) | Media | Se dibujan los puntos y luego se “vuelven a mirar” con una CNN, en vez de usar las coordenadas |
| [H10](#h10) | Baja | `predict_harp.py`: video fijo, preprocesamiento distinto al del entrenamiento |
| [H11](#h11) | Baja | `data_augmentation.py`: borra la carpeta de salida, sin semilla, técnicas distintas a las de la tesis |
| [H12](#h12) | Baja | `handtrack.py`: rutas con índices fijos, carpetas que no siempre se crean |
| [H13](#h13) | Baja | Reproducibilidad: sin semillas, dependencias en conflicto |
| [H14](#h14) | Baja | Extracción lenta: una imagen por llamada a `predict` |
| [H15](#h15) | Baja | Exceso de `print` que oculta la información útil |

## Estado de aplicación (6 de octubre de 2026)

Las recomendaciones de código se aplicaron en el orden de la [hoja de ruta](#hoja-de-ruta-sugerida). Las secciones H1 – H15 se conservan tal como se redactaron, porque describen la versión anterior y sirven para explicar en la tesis por qué cambiaron los resultados.

| Código | Estado | Dónde |
|---|---|---|
| H1 | **Corregido.** Inception V3 con `pooling='avg'`: 2048 rasgos por fotograma, sin capas aleatorias. Se guarda como `-features2048.npy` | `extract_features_harp.py` (`build_extractor`) |
| H2 | **Corregido.** Patrón `<nombre>[-_]NNNN.jpg`. Con los fotogramas actuales de `data/`, 103 de 415 secuencias tomaban fotogramas ajenos con el patrón viejo; con el nuevo, ninguna | `lspy_common.frame_pattern` |
| H3 | **Corregido.** Validación del 20 % de los originales de train (estratificada por clase y agrupada con sus copias); la prueba se evalúa una sola vez; `restore_best_weights=True` | `train_lstm_harp.py`, `lspy_common.grouped_train_val_split` |
| H4, H5 | **Ejecutado** (7 de octubre): IC95 y prueba binomial de cada modelo, y validación cruzada de 5 pliegues por persona con `personas_video.csv` (53,5 % ± 6,2 con `features2048`). La prueba resultó casi independiente del signante (ver 04 · 4.4) | [06 · 6.8](06_metricas_para_la_tesis.md#68-resultados-de-la-regeneración-completa-7-de-octubre-de-2026) |
| H6 | **Aplicado.** `--arch ligera` (por defecto, 1,17 M parámetros con 2048 entradas) u `--arch original` (35,6 M); tasa de aprendizaje 1e-4 por defecto (`--lr`) | `train_lstm_harp.py` |
| H7 | **Aplicado.** Se conservan los fotogramas sin manos (en negro) y cada video se remuestrea por tiempo a `--frames` fotogramas (150 por defecto); ya no hay relleno | `handtrack.py` (`process_video`) |
| H8 | **Aplicado.** MediaPipe procesa el cuadro completo con bordes negros (sin deformar) y se dibuja en 299 × 299 | `handtrack.py` (`detect_hands`) |
| H9 | **Aplicado.** `handtrack.py` guarda también las coordenadas (`-landmarks.npy`, 126 valores por fotograma, muñeca absoluta y resto relativo a la muñeca); se entrena con `--data-type landmarks` | `handtrack.py` (`landmarks_vector`) |
| H10 | **Corregido.** Video por argumento, mismo preprocesamiento que el entrenamiento (incluido el paso por JPEG), clases y representación leídas del `.json` del modelo, top-3 | `predict_harp.py` |
| H11 | **Corregido** y alineado con la tesis: rotación, zoom y eliminación aleatoria de fotogramas; espejo opcional (`--espejo`, usado en la regeneración del 7 de octubre); semilla; nunca borra la salida; registro `aumento_parametros.csv` | `data_augmentation.py` |
| H12 | **Corregido.** Rutas con `relpath` (cualquier profundidad), `splitext`, `makedirs(exist_ok=True)` para `sequences/`, `checkpoints/` y `logs/` | `handtrack.py` |
| H13 | **Corregido.** `set_seed(42)` en todos los scripts; `requirements.txt` sin `opencv-python`, `vidaug`, `imgaug` ni `imageio`, con `scikit-learn` | `lspy_common.set_seed`, `requirements.txt` |
| H14 | **Corregido.** Los fotogramas de un video se procesan en lotes de 32 | `extract_features_harp.py` (`extract_features`) |
| H15 | **Corregido.** Sin `print` de depuración; barras `tqdm` | todos |
| Varias semillas (tutor) | **Ejecutado.** Semillas 42 – 46: prueba 44,0 % ± 9,6; validación cruzada por persona 53,0 % ± 1,6. Incluye análisis de señas parecidas | `experimentos_semillas.sh`, `analisis_semillas.py`, [06 · 6.9](06_metricas_para_la_tesis.md#69-estabilidad-entre-semillas-y-señas-parecidas-7-de-octubre-de-2026) |
| Rol de Inception V3 | **Alineado con la tesis.** Nuevo `retrain_inception.py`: reentrena la última capa (softmax) de Inception V3 con los fotogramas de entrenamiento y guarda las probabilidades de cada seña por fotograma (`-probs.npy`) | `retrain_inception.py` |

El 7 de octubre de 2026 se regeneró todo el pipeline con el código corregido sobre `rawdata/` (aumento con espejo). Resultados en [06 · 6.8](06_metricas_para_la_tesis.md#68-resultados-de-la-regeneración-completa-7-de-octubre-de-2026). Los datos de la versión anterior se respaldaron en `../data_v1_inception_backup/`.

---

<a id="h1"></a>
## H1 · El extractor usa una capa densa aleatoria

**Dónde:** `extract_features_harp.py` líneas 147–162 y `predict_harp.py` líneas 29–43.

```python
x = base_model.output            # 8 × 8 × 2048
x = Flatten()(x)                 # 131 072 valores
predictions = Dense(data.class_limit, activation='softmax')(x)   # ← pesos aleatorios
model = Model(inputs=base_model.input, outputs=predictions)
```

**Qué pasa:**

1. La capa `Dense(10)` nunca se entrena: sus 1,3 millones de pesos se inicializan al azar. Lo que se guarda en cada `.npy` es una **proyección aleatoria de 131 072 valores a solo 10**, pasada por softmax. Toda la riqueza de Inception V3 (2048 rasgos por imagen) se comprime en 10 números arbitrarios. El comentario del código (“We'll extract features at the final pool layer”) describe lo que se quería hacer, pero no lo que se hace.
2. Esos 10 números no son “probabilidades de cada seña”, aunque sumen 1: la capa no sabe nada de señas.
3. **Cada ejecución crea pesos aleatorios nuevos.** Las características de entrenamiento se calcularon con una capa y `predict_harp.py` crea otra distinta. Para la LSTM, un video nuevo llega en un “idioma” diferente del que aprendió, así que la predicción sobre videos nuevos no es fiable aunque el entrenamiento haya ido bien. Lo mismo ocurre si se borra parte de `data/sequences/` y se vuelve a extraer.

**Por qué importa para la tesis:** es la explicación más probable de las exactitudes bajas del capítulo 6 (18 % – 30 % con 10 clases): el modelo trabaja con una representación muy empobrecida.

**Cómo corregirlo** (cambio pequeño, no requiere entrenar Inception):

```python
# extract_features_harp.py
base_model = InceptionV3(weights='imagenet', include_top=False,
                         pooling='avg', input_shape=(299, 299, 3))   # salida: 2048 por imagen
model = base_model

# ... dentro del bucle, procesar los 150 fotogramas de una vez:
batch = np.stack([preprocess_input(Img.img_to_array(Img.load_img(f, target_size=(299, 299))))
                  for f in frames])
sequence = model.predict(batch, batch_size=32, verbose=0)           # (150, 2048)
np.save(path, sequence)
```

- Guardar con otro sufijo (por ejemplo `'-features2048'`) para no mezclar con los `.npy` viejos, y usar `data_type='features2048'` al cargar.
- En `train_lstm_harp.py`, reemplazar `input_shape=(150, 10)` por `input_shape=X.shape[1:]`.
- En `predict_harp.py`, usar exactamente el mismo `base_model` con `pooling='avg'` (sin `Dense`). Al ser determinista, entrenamiento y predicción quedan alineados.
- Variante más ambiciosa: ajuste fino (*fine-tuning*) de las últimas capas de Inception con los dibujos de manos, guardando el modelo resultante y reutilizándolo en la predicción.

`evaluate_metrics.py` detecta este problema y lo avisa (“¿vectores softmax? True”).

---

<a id="h2"></a>
## H2 · Secuencias que mezclan fotogramas de varios videos

**Dónde:** `extract_features_harp.py`, `get_frames_for_sample`:

```python
images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
```

**Qué pasa:** el patrón `<nombre>*jpg` también encuentra los fotogramas de **cualquier video cuyo nombre empiece igual**. Como las copias aumentadas se llaman `<original>_0`, `<original>_1`…, para el original `agosto_cr5` el patrón devuelve sus 150 imágenes **y las de sus 4 copias** (750 en total). Luego `rescale_list(…, 150)` toma una de cada 5: la “secuencia” del original queda armada con 30 fotogramas de cada uno de 5 videos distintos, intercalados en un orden que no respeta el tiempo.

Además, el relleno del original se llama `agosto_cr5_0141.jpg`, que **coincide con el patrón de la copia** `agosto_cr5_0`, así que algunas copias también reciben fotogramas ajenos.

**Alcance estimado:** los 79 originales de `train/` (uno de cada cinco ejemplos de entrenamiento) y parte de sus copias. En `test/` no ocurre, porque allí no hay copias. Cifra exacta: `evaluate_metrics.py` la reporta en “colisiones de nombre” y en la columna `fotogramas_ajenos` de `cobertura_manos_por_video.csv`.

**Cómo corregirlo** sin volver a ejecutar `handtrack.py` (acepta tanto `-NNNN` como el relleno `_NNNN`):

```python
pattern = glob.escape(filename) + '[-_][0-9][0-9][0-9][0-9].jpg'
images = sorted(glob.glob(os.path.join(path, pattern)))   # primero los reales (-), luego el relleno (_)
```

Y, para la próxima vez que se ejecute `handtrack.py`, usar el mismo separador en el relleno:

```python
'{}-{}.jpg'.format(filename_no_ext, str.rjust(str(i), 4, '0'))   # en lugar de '{}_{}.jpg'
```

Después de corregir, **borrar `data/sequences/*.npy`** y volver a extraer.

---

<a id="h3"></a>
## H3 · El conjunto de prueba se usa como validación

**Dónde:** `train_lstm_harp.py`, `model.fit(..., validation_data=(X_test, y_test))`, con `ModelCheckpoint(save_best_only=True)` y `EarlyStopping` vigilando `val_loss`.

**Qué pasa:** el conjunto de prueba decide cuándo parar y qué pesos guardar. La exactitud que se mide después sobre ese mismo conjunto es **optimista**: el modelo fue elegido por rendir bien en él. El marco teórico (sección 4.4) explica que la prueba “se usa una sola vez, como un examen”.

**Cómo corregirlo:** separar una validación **desde train, agrupando por video original** para que las copias de un mismo video no queden repartidas:

```python
from sklearn.model_selection import GroupShuffleSplit
from evaluate_metrics import original_stem

train_rows, test_rows = data.split_train_test()
groups = [original_stem(r[2]) for r in train_rows]
tr_idx, va_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                      .split(train_rows, groups=groups))
# X_tr, y_tr = X[tr_idx], y[tr_idx];  X_val, y_val = X[va_idx], y[va_idx]
model.fit(X_tr, y_tr, validation_data=(X_val, y_val), ...)
model.evaluate(X_test, y_test)      # una sola vez, al final
```

Además, guardar el **mejor** modelo y no el de la última época: `EarlyStopping(patience=30, restore_best_weights=True)` antes de `model.save(...)`.

---

<a id="h4"></a>
## H4 · Conjunto de prueba muy chico

20 videos (2 por clase). Cada video vale 5 puntos de exactitud, y con 2 ejemplos por clase la exhaustividad de cada seña solo puede valer 0 %, 50 % o 100 %. Un intervalo de confianza del 95 % para una exactitud observada del 30 % va aproximadamente del 15 % al 52 % (intervalo de Wilson).

**Recomendaciones:**

- Informar siempre la exactitud **con su intervalo de confianza** (`evaluate_metrics.py` lo calcula por bootstrap) y con el número de aciertos (p. ej. “6 de 20”).
- Complementar con **validación cruzada agrupada por video original** sobre los 99 videos reales: `python evaluate_metrics.py --cv 5 --cv-incluir-test --cv-solo-originales`. Así cada video real se usa una vez para evaluar y la estimación es mucho más estable.
- Comparar con el azar mediante la prueba binomial que reporta el script.

---

<a id="h5"></a>
## H5 · Evaluación dependiente del signante

Las personas de test aparecen también en train (ver [04 · 4.4](04_conjunto_de_datos.md#44-personas-que-signan-y-tipo-de-evaluación)). No hay fuga de videos, pero el resultado mide el reconocimiento de personas **conocidas**.

**Recomendaciones:**

- Declararlo explícitamente en la metodología y en las conclusiones.
- Si hay al menos 3 personas, hacer una evaluación **dejando una persona fuera** (`--cv-por-persona` con `docs/personas_plantilla.csv` completo) y presentarla junto a la evaluación actual. La diferencia entre ambas es un resultado interesante en sí mismo (el capítulo 3 cita la caída de AUTSL de 95,95 % a 62,02 %).

---

<a id="h6"></a>
## H6 · Modelo sobredimensionado

La primera LSTM tiene 2048 unidades para entradas de 10 valores: **16,9 millones de parámetros** solo en esa capa, 18,9 M en total, para ≈ 400 secuencias (79 grabaciones reales en train). Es el escenario clásico de sobreajuste descrito en el marco teórico. A eso se suma una tasa de aprendizaje de 1e-5, muy baja para Adam (lo habitual es 1e-3 a 1e-4), que hace que el modelo aprenda muy despacio en 100 épocas.

**Recomendación:** una arquitectura liviana como punto de comparación (está implementada como `--cv-arch ligera` en `evaluate_metrics.py`):

```python
model = Sequential([
    Input(shape=X.shape[1:]),
    LSTM(128, return_sequences=True, dropout=0.3),
    LSTM(64, dropout=0.3),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(data.classes), activation='softmax'),
])
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
```

Con entradas de 2048 (tras corregir H1) tiene ≈ 1,2 M parámetros; con entradas de 10, ≈ 125 mil. Presentar en la tesis una tabla “arquitectura original vs. liviana” es un experimento sencillo y muy informativo.

---

<a id="h7"></a>
## H7 · Tratamiento temporal: descarte, relleno y fps

- Los fotogramas **sin manos se descartan** (no se guardan). Si MediaPipe pierde la mano a mitad de la seña, la secuencia salta en el tiempo sin que el modelo lo sepa.
- La secuencia se completa hasta 150 **repitiendo el último dibujo**. En videos cortos o con mala detección, buena parte de la entrada es relleno idéntico.
- Con videos de 60 fps y de 30 fps mezclados, 150 fotogramas representan 2,5 s o 5 s según el caso: la misma seña tiene “velocidades” distintas para el modelo.

**Recomendación:** guardar todos los fotogramas (negros si no hay mano) y **remuestrear por tiempo** a una longitud fija:

```python
idx = np.linspace(0, n_frames - 1, num=T).round().astype(int)   # T = 40 o 60, por ejemplo
frames = [frames[i] for i in idx]
```

Con T = 40 – 60 la secuencia es más corta, el entrenamiento más rápido y los videos de 30 y 60 fps quedan comparables. Informar en la tesis el porcentaje de relleno por clase (`tabla_cobertura_manos.tex`).

---

<a id="h8"></a>
## H8 · Deformación de la imagen antes de MediaPipe

`cv2.resize(frame, (299, 299))` convierte un cuadro vertical de 720 × 1280 en uno cuadrado: la imagen queda comprimida horizontalmente casi a la mitad, y MediaPipe recibe manos deformadas y de baja resolución (299 px de alto para todo el cuerpo). Esto puede bajar la tasa de detección.

**Recomendación:** detectar sobre el cuadro original (o sobre un cuadro con bordes negros que respete la proporción) y solo después dibujar en el lienzo de 299 × 299:

```python
results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))   # cuadro original
h, w = frame.shape[:2]; s = max(h, w)                              # lienzo cuadrado sin deformar
canvas = np.zeros((s, s, 3), np.uint8)
# ...dibujar los puntos con un desplazamiento ((s - w) // 2, (s - h) // 2) y luego:
img = cv2.resize(canvas, (299, 299))
```

---

<a id="h9"></a>
## H9 · Representación: dibujar los puntos para volver a leerlos con una CNN

MediaPipe ya entrega las coordenadas (x, y, z) de 21 puntos por mano. El pipeline las dibuja como imagen y luego usa Inception V3 (entrenada con fotos de objetos) para “leer” ese dibujo. Es un camino indirecto y costoso: Inception no fue entrenada con esqueletos de manos, y la etapa es la más lenta del sistema.

**Recomendación (experimento comparativo para la tesis):** usar directamente las coordenadas como entrada de la LSTM. Es lo que hacen los trabajos citados en el estado del arte (Samaan et al. 2022, Selvaraj et al. 2022):

```python
def landmarks_vector(results):
    """126 valores por fotograma: 2 manos × 21 puntos × (x, y, z); ceros si falta una mano."""
    out = np.zeros((2, 21, 3), np.float32)
    if results.multi_hand_landmarks:
        for hand, info in zip(results.multi_hand_landmarks, results.multi_handedness):
            k = 0 if info.classification[0].label == 'Left' else 1
            out[k] = [[p.x, p.y, p.z] for p in hand.landmark]
    return out.ravel()
```

Conviene normalizar respecto de la muñeca (restar el punto 0) para independizarse de la posición en el cuadro. El resultado es una secuencia (T, 126), mucho más liviana que (150, 2048), y permite comparar en la tesis “dibujo + Inception V3” contra “coordenadas directas”.

---

<a id="h10"></a>
## H10 · `predict_harp.py`

- El video está fijo en el código (`image_name = 'ROJO.mp4'`); conviene recibirlo por argumento (`argparse`).
- En el entrenamiento, los dibujos pasan por JPEG (compresión con pérdida) y se leen con `load_img`; en la predicción se usan directamente los arreglos. Son pequeñas diferencias de dominio; lo ideal es compartir una única función de preprocesamiento.
- El “`class_name` por fotograma” se calcula con la capa aleatoria (H1) y no tiene significado.
- `seq_lenght = 40` no se usa; `classes[i].split('/')[2]` falla en Windows (usar `os.path.basename`).
- Las clases se deducen de las carpetas de `data/train`; si falta una, los índices se desplazan. Mejor guardar la lista de clases junto al modelo (por ejemplo `classes.json`).

---

<a id="h11"></a>
## H11 · `data_augmentation.py`

- **Borra la carpeta de salida** si existe (`shutil.rmtree`). Un error en la ruta puede eliminar videos originales. Es lo que ocurre, por diseño, cada vez que se vuelve a ejecutar sobre `trainaug/`.
- Rotación aleatoria **sin semilla**: no es reproducible. Agregar `random.seed(...)` o guardar el ángulo usado en el nombre o en un CSV.
- **Espejo horizontal:** convierte una seña diestra en zurda. Es una aumentación razonable si la LSPy admite ambos, pero conviene justificarlo en la tesis (o consultarlo con un intérprete), porque en algunas señas la mano dominante o la orientación puede cambiar el significado.
- **Diferencia con la tesis:** el capítulo 6 menciona “rotación de ángulos, zoom y eliminación de fotogramas aleatorios”. El código actual solo hace **espejo y rotación**. Si zoom y eliminación de fotogramas se usaron en otra versión, conviene indicarlo; si no, corregir el texto.
- El ruido está desactivado (`noise_value = 0`) y hay importaciones sin uso (`vidaug`, `imageio`, `Pool`).
- La variable global `clip_no` se comparte entre hilos; funciona porque el `with` espera a que terminen, pero es frágil. Pasar el índice del video como argumento.

---

<a id="h12"></a>
## H12 · `handtrack.py`

- `get_video_parts` usa `parts[1]`, `parts[2]`, `parts[3]`: solo funciona con `-i` de un nivel. Mejor: `os.path.relpath(video_path, vid_dir).split(os.sep)`.
- `sequences/` y `checkpoints/` solo se crean si `out_dir` no existía; `logs/` nunca. Usar `os.makedirs(..., exist_ok=True)` para todas.
- `filename.split('.')[0]` corta en el primer punto; con nombres como `video.v2.mp4` falla. Usar `os.path.splitext`.
- El espejo horizontal (`cv2.flip(image, 1)`) se aplica a todos los videos; no afecta el entrenamiento mientras se haga igual en la predicción, pero conviene documentarlo.

---

<a id="h13"></a>
## H13 · Reproducibilidad y dependencias

- No hay semillas. Agregar al inicio de cada script: `tf.keras.utils.set_random_seed(42)` (fija Python, NumPy y TF).
- `requirements.txt` instala `opencv-python` y `opencv-contrib-python` a la vez (conflicto). Dejar uno.
- Falta `scikit-learn` para las métricas (`pip install scikit-learn==1.5.2`).
- Registrar en la tesis las versiones exactas (Python 3.10, TF 2.15.1, MediaPipe 0.10.14) y el hardware usado.

---

<a id="h14"></a>
## H14 · Extracción lenta

`model.predict(x)` se llama una vez por imagen: 150 llamadas por video, cada una con su sobrecarga. Procesar los 150 fotogramas en un solo lote (ver el código de H1) suele acelerar la extracción varias veces, sobre todo en GPU.

---

<a id="h15"></a>
## H15 · Mensajes de consola

`DataSet` imprime el contenido completo de `data_file.csv` varias veces y cada búsqueda de archivo; `handtrack.py` imprime cada número de fotograma. Reemplazar por `logging` con nivel configurable o por la barra `tqdm` que ya se usa.

---

## Hoja de ruta sugerida

| Prioridad | Acción | Esfuerzo | Impacto |
|---|---|---|---|
| 1 | Corregir H2 (patrón del `glob`) y H1 (`pooling='avg'`, 2048 dimensiones); borrar `data/sequences` y volver a extraer | Bajo (≈ 10 líneas) | Muy alto: datos correctos y representación rica |
| 2 | Separar validación desde train agrupando por video original (H3); `restore_best_weights=True` | Bajo | Alto: resultados honestos |
| 3 | Ejecutar `evaluate_metrics.py` sobre el modelo actual (línea base) y sobre el corregido | Bajo | Alto: material para el capítulo 6 |
| 4 | Validación cruzada de 5 pliegues agrupada por video (H4) y, si hay datos de personas, por persona (H5) | Medio (tiempo de cómputo) | Alto |
| 5 | Comparar arquitectura original vs. liviana (H6) | Bajo | Medio |
| 6 | Remuestreo temporal y detección sin deformar (H7, H8); volver a ejecutar `handtrack.py` | Medio | Medio |
| 7 | Experimento con coordenadas de MediaPipe en vez de dibujo + Inception (H9) | Medio | Potencialmente alto; buen aporte para la discusión |
| 8 | Limpiezas de H10 – H15 | Bajo | Calidad y reproducibilidad |

Cada cambio se puede presentar en la tesis como un experimento con su tabla de métricas, lo que además enriquece el capítulo de resultados.

## Puntos de la tesis a alinear con el código

| Tema | Tesis (versión actual) | Código anterior | Código actual / qué hacer en la tesis |
|---|---|---|---|
| Rol de Inception V3 | “extrae características y probabilidades de pertenencia a cada etiqueta”; README: “Retraining” | Inception sin reentrenar + `Dense(10)` aleatoria; las salidas no son probabilidades de señas (H1) | **Alineado.** Inception V3 (ImageNet, congelada) extrae 2048 características por fotograma y `retrain_inception.py` reentrena su última capa (transferencia de aprendizaje, como el `retrain.py` de TensorFlow) para obtener la probabilidad de cada seña por fotograma. La LSTM puede usar las características (`features2048`) o las probabilidades (`probs`); conviene informar ambas. Ver texto sugerido abajo |
| Técnicas de aumento | rotación, zoom, eliminación de fotogramas | espejo horizontal + rotación ±30° (H11) | **Alineado.** Rotación (±30°), zoom (85 % – 115 %), eliminación aleatoria de hasta el 20 % de los fotogramas y, con `--espejo` (usado en la regeneración del 7 de octubre de 2026), espejo horizontal en las copias 0 y 2. Agregar el espejo al texto de la tesis |
| Tamaño del conjunto | 10 originales × 5 = 50 por etiqueta, 500 en total | 7–9 originales por clase en train (79) + 2 en test (20); 395 en train con copias | Con `rawdata/`: 80 originales en train (7 – 9 por seña) + 20 en test; 400 secuencias de train con copias. Corregir el texto con estas cifras |
| Validación | — | El conjunto de prueba se usa como validación (H3) | Validación separada desde train, agrupada por video original. Describirlo en la metodología |
| Tipo de evaluación | — | Mismas personas en train y test (H5) | Con `rawdata/` y `personas_video.csv`, la prueba es **casi independiente del signante** (solo `marzo_cr1_2` comparte persona y seña con train). Declararlo en la metodología e informar también la validación cruzada por persona |
| Métrica | “precisión” | Lo que se mide es **exactitud** | Usar “exactitud” |
| Arquitectura LSTM | “una capa LSTM … otra capa de LSTM” (2 capas) | 3 capas LSTM (2048, 256, 128) + Dense(512) intermedia | La arquitectura por defecto (`ligera`) tiene **2 capas LSTM** (128 y 64), como dice la tesis. Si se informa la original, corregir el texto |

### Texto sugerido: rol de Inception V3

> Inception V3, preentrenada con ImageNet, se utiliza mediante transferencia de aprendizaje. Sus capas convolucionales se mantienen congeladas y actúan como extractor de características: para cada fotograma, el promedio global de la última capa convolucional produce un vector de 2048 valores. Sobre ese vector se reentrena una nueva capa final de clasificación (softmax) con los fotogramas del conjunto de entrenamiento, etiquetados con la seña del video al que pertenecen. Así, para cada fotograma se obtienen tanto sus características como la probabilidad de pertenencia a cada una de las 10 señas. La secuencia de estos vectores es la entrada de la red LSTM.

Por qué solo la última capa: con 79 grabaciones de entrenamiento (≈ 12 000 fotogramas originales, muy parecidos entre sí dentro de cada video), ajustar millones de pesos convolucionales sobreajustaría; una capa softmax tiene 20 490 parámetros y se entrena en segundos sin GPU. Para no inflar los resultados, la capa se entrena sin los videos de validación ni de prueba, y las probabilidades de los videos de entrenamiento se calculan fuera de pliegue (cada video las recibe de una capa que no lo vio).

### Texto sugerido: aumento de datos

> Para ampliar el conjunto de entrenamiento se generaron cuatro variaciones de cada video original de entrenamiento. Cada variación aplica a todos sus fotogramas una rotación aleatoria de entre −30° y 30° y un zoom aleatorio de entre 85 % y 115 %, y elimina al azar hasta el 20 % de los fotogramas, lo que simula variaciones en la velocidad de ejecución de la seña. Además, dos de las cuatro variaciones se espejan horizontalmente, lo que simula a una persona que signa con la otra mano como dominante. Los parámetros se obtienen de un generador con semilla fija y se registran para cada copia. El conjunto de prueba no se aumentó.

Conviene justificar el espejo (o consultarlo con un intérprete de LSPy): en las señas cuyo significado no depende de la mano dominante es una variación válida; si alguna seña cambia de significado al espejarse, habría que excluirla del espejo.
