# 2. Instalación y guía de uso paso a paso

## 2.1 Requisitos

- Linux (probado en Ubuntu; las rutas usan `/`).
- **Python 3.10** (3.9 – 3.11 funcionan con TensorFlow 2.15; 3.12 o superior **no**).
- ~2 GB libres para el entorno y varios GB para los fotogramas `.jpg` que genera `handtrack.py` (≈ 150 imágenes por video).
- GPU opcional. Sin GPU, la extracción con Inception V3 es la etapa más lenta (ver la latencia que mide `evaluate_metrics.py --latencia-inception`).

## 2.2 Crear el entorno

```bash
cd ~/Documents/LSPY/inception-v3
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notas sobre `requirements.txt`:

- Solo se instala `opencv-contrib-python` (el que pide MediaPipe). Si un entorno viejo tiene también `opencv-python`, desinstale ambos y vuelva a instalar solo `opencv-contrib-python`.
- `jax`, `jaxlib` y `sounddevice` llegan como dependencias de MediaPipe; el proyecto no los usa directamente.
- `scikit-learn` se usa en `evaluate_metrics.py`.

## 2.3 Estructura esperada de los videos

```
rawdata/                     # originales
├── train/
│   ├── agosto/   agosto_cr3.mp4, agosto_marcelo.mp4, …  (8 por seña)
│   ├── broma/
│   └── …         (10 carpetas, una por seña)
└── test/
    ├── agosto/   agosto_cr1.mp4, agosto_cr2.mp4
    └── …

rawdata_aug/                 # salida del paso 1, entrada del paso 2
├── train/<clase>/  originales + copias <nombre>_0 … _3.mp4 + aumento_parametros.csv
└── test/<clase>/   copia de rawdata/test (sin aumentar)
```

- El nombre de la carpeta es la etiqueta de la clase.
- `handtrack.py` solo recorre `train/` y `test/`. Cualquier otra carpeta (por ejemplo `trainaug/`) se ignora.

## 2.4 Ejecución del pipeline

Todos los comandos se ejecutan desde la raíz del repositorio. Si `data/` tiene fotogramas o `.npy` de la versión anterior del código, conviene empezar con una carpeta limpia (por ejemplo, renombrar `data/` a `data_v1/`).

### Paso 1 — Aumento de datos (opcional)

```bash
python data_augmentation.py \
    --main-folder-path rawdata/train \
    --output-folder-path rawdata_aug/train \
    --max-clips 4 --espejo --copiar-originales
cp -r rawdata/test rawdata_aug/test
```

- `--main-folder-path` puede ser una partición con una subcarpeta por clase (como arriba) o la carpeta de una sola clase.
- Por cada video genera `--max-clips` copias `<nombre>_<i>.mp4` con **rotación** (±30°, `--max-rotacion`), **zoom** (85 % – 115 %, `--max-zoom`) y **eliminación aleatoria de fotogramas** (hasta 20 %, `--max-eliminacion`). Con `--espejo`, además, las copias de índice par se espejan.
- Los videos cuyo nombre indica que ya son copias (`<nombre>_<i>`) se omiten, salvo con `--incluir-aumentados`. Así, aunque la carpeta de entrada tenga copias de una versión anterior, solo se leen los originales y las copias viejas no pasan a la salida.
- `--copiar-originales` copia también cada original a la salida, que queda lista para `handtrack.py` (originales + copias nuevas).
- Es reproducible (`--semilla`, 42 por defecto) y registra los parámetros de cada copia en `<salida>/aumento_parametros.csv`.
- **Nunca borra** la carpeta de salida; las copias que ya existen se saltan, salvo con `--sobrescribir`.
- Aumente **solo** el conjunto de entrenamiento, nunca el de prueba (ver fuga de datos en el marco teórico, sección 4.9).

### Paso 2 — Detección de manos

```bash
python handtrack.py -i rawdata_aug -o data            # 150 fotogramas por video
python handtrack.py -i rawdata_aug -o data -n 60      # alternativa: 60 (usar luego --seq-length 60)
```

- Recorre `train/` y `test/` de la carpeta de entrada (con cualquier ruta, relativa o absoluta).
- MediaPipe procesa **todos** los cuadros, completos y sin deformar; luego cada video se **remuestrea por tiempo** a `-n` fotogramas equiespaciados. Los cuadros sin manos quedan en negro (no se descartan ni se rellena).
- Resultado:
  - `data/train|test/<Clase>/<video>-NNNN.jpg`: dibujos de las manos, 299 × 299.
  - `data/sequences/<partición>/<video>-<n>-landmarks.npy`: coordenadas de MediaPipe (n × 126).
  - `data/data_file.csv`: `partición, clase, video, n, fotogramas con mano, fotogramas del video, fps`.
- Si se vuelve a ejecutar sobre la misma salida, borra los fotogramas y `.npy` anteriores de cada video que procesa.
- Tarda unos 4 s por video sin GPU (≈ 30 min para los 415 videos).

### Paso 3 — Extracción de características con Inception V3

```bash
python extract_features_harp.py                       # agregar --seq-length 60 si se usó -n 60
```

- Guarda `data/sequences/<partición>/<video>-150-features2048.npy` (150 × 2048): el promedio global de la última capa convolucional de Inception V3 (ImageNet) para cada fotograma.
- Si el `.npy` ya existe, lo salta.
- La primera vez descarga los pesos de ImageNet (~90 MB) a `~/.keras/models/`.
- Unos 50 fotogramas por segundo sin GPU (≈ 20 min para 415 videos × 150 fotogramas).

### Paso 4 — Reentrenamiento de la última capa de Inception V3 (opcional)

```bash
python retrain_inception.py
```

- Entrena una capa softmax nueva sobre las 2048 características de cada fotograma de entrenamiento (sin los videos de validación ni de prueba) y la guarda en `data/inception_head.keras`.
- Guarda `data/sequences/<partición>/<video>-150-probs.npy` (150 × 10): la probabilidad de cada seña por fotograma. Las de los videos de entrenamiento se calculan fuera de pliegue (`--folds 5`).
- Informa la exactitud de Inception V3 sola (por fotograma y por video, promediando), útil como línea base sin LSTM.
- Tarda segundos o pocos minutos: no vuelve a pasar las imágenes por Inception.

### Paso 5 — Entrenamiento de la LSTM

```bash
python train_lstm_harp.py                                         # features2048, arquitectura ligera
python train_lstm_harp.py --data-type probs --output lstm_probs     # probabilidades de Inception reentrenada
python train_lstm_harp.py --data-type landmarks --output lstm_lm    # coordenadas de MediaPipe (H9)
python train_lstm_harp.py --arch original --output lstm_original   # arquitectura de 3 capas (H6)
```

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--data-type` | `features2048` | `features2048`, `probs`, `landmarks` o `features` (versión anterior) |
| `--arch` | `ligera` | `ligera`: LSTM(128) → LSTM(64) → Dense(64); `original`: LSTM(2048) → Dense(512) → LSTM(256) → LSTM(128) |
| `--lr` | `1e-4` | Tasa de aprendizaje de Adam |
| `--epochs`, `--patience` | 100, 30 | Épocas máximas y paciencia de la parada temprana |
| `--val-frac` | 0.2 | Fracción de originales de train para validación (cada uno con sus copias) |
| `--seq-length`, `--class-limit`, `--seed` | 150, 10, 42 | |
| `--split-seed` | 42 | Semilla de la separación train/validación (independiente de `--seed`) |
| `--clases` | todas | Entrenar solo con algunas señas, p. ej. `--clases marzo sabado jugar nombre miercoles` |
| `--sin-checkpoints` | no | No guardar un `.hdf5` por época |
| `--output` | `lstm_senha_model` | Carpeta del modelo (SavedModel) |

Guarda:

- `<output>/` (modelo con los **mejores** pesos según validación) y `<output>.json` (clases, representación, arquitectura, videos de validación y exactitud de prueba).
- `data/checkpoints/lstm-<data_type>-<arch>.<época>-<val_loss>.hdf5`.
- `data/logs/lstm-<data_type>-<arch>-training-<timestamp>.log` (CSV por época) y `data/logs/lstm-<data_type>-<arch>/` (TensorBoard).

La exactitud de prueba se calcula una sola vez, al final, y se imprime como “Prueba: exactitud … (aciertos de total)”.

### Paso 6 — Predicción sobre un video

```bash
python predict_harp.py ROJO.mp4
python predict_harp.py ROJO.mp4 --model lstm_probs --top 5
```

- Aplica exactamente los mismos pasos que el entrenamiento (detección sin deformar, remuestreo, paso por JPEG, Inception V3 y, si corresponde, la capa reentrenada), según lo que indique `<modelo>.json`.
- Imprime las señas más probables y la ganadora.

### Paso 7 — Métricas para la tesis

```bash
python evaluate_metrics.py --model lstm_senha_model --raw-dir rawdata_aug --signers-csv personas_video.csv
```

La representación (`--data-type`) se toma del `.json` del modelo. Detalles en [06 · Métricas](06_metricas_para_la_tesis.md).

### Todo junto

```bash
./pipeline_completo.sh     # pasos 1 a 7 y la batería de semillas, desde rawdata/ (~12 h sin GPU)
```

Salta las etapas ya hechas (`rawdata_aug/`, `data/data_file.csv`, `data/inception_head.keras`, modelos y métricas existentes). Para empezar de cero, mover o borrar esas carpetas antes.

### Paso 8 — Repetición con varias semillas y análisis de señas parecidas (opcional)

```bash
./experimentos_semillas.sh          # semillas 42 a 46; SEMILLAS="42 43" ./experimentos_semillas.sh para elegir otras
python analisis_semillas.py         # media ± desvío entre semillas y análisis de pares de señas parecidas
```

- Entrena las variantes con cada semilla (modelos en `modelos_semillas/`, sin checkpoints por época), repite la validación cruzada por persona y entrena modelos solo con señas no parecidas.
- Tarda unas 9 horas sin GPU, la mayor parte en la arquitectura `original`. Si se interrumpe, al volver a ejecutarlo salta lo que ya terminó.
- Los pares de señas parecidas se definen en `PARES`, al comienzo de `analisis_semillas.py`.
- Resultados e interpretación en [06 · 6.9](06_metricas_para_la_tesis.md#69-estabilidad-entre-semillas-y-señas-parecidas-8-de-octubre-de-2026).

## 2.5 Utilidad suelta

`loadpicklefileanddisplay.py` muestra el contenido de un archivo `.pickle` (`python loadpicklefileanddisplay.py archivo.pickle`). Hoy ningún paso del pipeline genera pickles: es un resto de una versión anterior.

## 2.6 Problemas frecuentes

| Síntoma | Causa probable | Solución |
|---|---|---|
| `FileNotFoundError: data/data_file.csv` | No se ejecutó `handtrack.py` o se usó otro `-o` | Ejecutar el paso 2 con `-o data` |
| `FileNotFoundError: Can't find sequence data/sequences/…` | Falta el `.npy` de esa representación | Ejecutar el paso 3 (`features2048`), el 4 (`probs`) o el 2 (`landmarks`) con el mismo `--seq-length` |
| `AssertionError` en `rescale_list` o el aviso “tiene N fotogramas (< 150)” | Se ejecutó `handtrack.py -n` con un valor menor que `--seq-length` | Usar el mismo número en ambos |
| `module 'cv2' has no attribute …` | Conflicto entre `opencv-python` y `opencv-contrib-python` | Desinstalar ambos e instalar solo `opencv-contrib-python` |
| `load_model` falla con un directorio SavedModel | Keras 3 (TF ≥ 2.16) | Usar TF 2.15 o guardar como `.keras` / `.h5` |
| Muchos videos con pocos fotogramas con mano | MediaPipe no detectó manos | Revisar la columna 5 de `data_file.csv` o `cobertura_manos_por_video.csv` de `evaluate_metrics.py` |
