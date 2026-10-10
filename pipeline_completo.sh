#!/bin/bash
# Ejecuta todo el pipeline desde los videos de rawdata/ hasta las métricas de la tesis:
#   1. aumento de datos (rotación, zoom, eliminación de fotogramas y espejo) -> rawdata_aug/
#   2. detección de manos (handtrack.py)                                     -> data/
#   3. características de Inception V3 (extract_features_harp.py)
#   4. reentrenamiento de la última capa de Inception V3 (retrain_inception.py)
#   5. modelos principales (semilla 42) y sus métricas                       -> lstm_*/, metricas/<modelo>/
#   6. repetición con semillas 42 a 46 y señas no parecidas (experimentos_semillas.sh)
#   7. resumen entre semillas y análisis de señas parecidas (analisis_semillas.py)
# Registros de cada etapa en data/logs_pipeline/. Las etapas ya hechas se saltan al relanzarlo.
# Unas 12 horas sin GPU.
set -e
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2 GLOG_minloglevel=2
PY=${PY:-.venv/bin/python}
L=data/logs_pipeline
mkdir -p $L

if [ ! -f rawdata_aug/train/aumento_parametros.csv ]; then
  echo "== 1 aumento $(date +%H:%M)"
  $PY data_augmentation.py --main-folder-path rawdata/train --output-folder-path rawdata_aug/train \
    --max-clips 4 --espejo --copiar-originales > $L/1_aumento.log 2>&1
  mkdir -p rawdata_aug/test && cp -r rawdata/test/. rawdata_aug/test/
fi
if [ ! -f data/data_file.csv ]; then
  echo "== 2 detección de manos $(date +%H:%M)"
  $PY handtrack.py -i rawdata_aug -o data > $L/2_handtrack.log 2>&1
fi
echo "== 3 extracción $(date +%H:%M)"
$PY extract_features_harp.py > $L/3_extraccion.log 2>&1
if [ ! -f data/inception_head.keras ]; then
  echo "== 4 reentrenamiento de Inception V3 $(date +%H:%M)"
  $PY retrain_inception.py > $L/4_reentrenamiento.log 2>&1
  grep "Inception V3 reentrenada" $L/4_reentrenamiento.log
fi

set +e
for cfg in "lstm_senha_model --data-type features2048" \
           "lstm_probs --data-type probs" \
           "lstm_landmarks --data-type landmarks --epochs 300 --patience 50" \
           "lstm_original --data-type features2048 --arch original"; do
  set -- $cfg
  modelo=$1; shift
  if [ ! -d $modelo ]; then
    echo "== 5 entrenamiento $modelo $(date +%H:%M)"
    $PY train_lstm_harp.py "$@" --sin-checkpoints --output $modelo > $L/5_train_$modelo.log 2>&1 || echo "FALLO $modelo"
    grep "Prueba:" $L/5_train_$modelo.log
  fi
  if [ ! -f metricas/$modelo/resumen.json ]; then
    $PY evaluate_metrics.py --model $modelo --raw-dir rawdata_aug --signers-csv personas_video.csv \
      --out-dir metricas/$modelo > $L/6_metricas_$modelo.log 2>&1 || echo "FALLO métricas $modelo"
  fi
done

echo "== 6 semillas $(date +%H:%M)"
./experimentos_semillas.sh
echo "== 7 análisis $(date +%H:%M)"
$PY analisis_semillas.py > $L/7_analisis_semillas.log 2>&1 && echo "Resumen en metricas/analisis_semillas/resumen_semillas.md"
echo "== fin $(date +%H:%M)"
