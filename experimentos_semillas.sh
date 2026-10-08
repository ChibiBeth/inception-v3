#!/bin/bash
# Repite el entrenamiento y la evaluación con varias semillas (42 a 46) y entrena modelos
# solo con señas no parecidas. Se puede interrumpir y volver a lanzar: salta lo ya hecho.
# Resultados: metricas/semillas/<experimento>/s<semilla>/ y metricas/semillas_cv/<experimento>/s<semilla>/
# Resumen: python analisis_semillas.py
cd "$(dirname "$0")"
export TF_CPP_MIN_LOG_LEVEL=2 GLOG_minloglevel=2
PY=${PY:-.venv/bin/python}
SEMILLAS=${SEMILLAS:-"42 43 44 45 46"}
M=modelos_semillas
L=data/logs_pipeline/semillas
mkdir -p $M $L
# Pares de señas parecidas: (marzo, rojo) (sabado, agosto) (jugar, broma) (nombre, martes) (miercoles, soltero)
GRUPO_A="marzo sabado jugar nombre miercoles"   # una seña de cada par
GRUPO_B="rojo agosto broma martes soltero"      # la otra

# Modelos de semilla 42 ya entrenados por el pipeline principal
declare -A EXISTENTE=([features2048_ligera]=lstm_senha_model [probs_ligera]=lstm_probs
                      [landmarks_ligera]=lstm_landmarks [features2048_original]=lstm_original)

entrenar() {  # experimento semilla argumentos...
  local exp=$1 s=$2; shift 2
  local modelo=$M/${exp}_s$s
  if [ "$s" = 42 ] && [ -n "${EXISTENTE[$exp]}" ]; then modelo=${EXISTENTE[$exp]}; fi
  if [ ! -d "$modelo" ]; then
    echo "== entrenar $exp s$s $(date +%H:%M)"
    $PY train_lstm_harp.py "$@" --seed $s --sin-checkpoints --output $modelo > $L/train_${exp}_s$s.log 2>&1 \
      || { echo "FALLO entrenar $exp s$s"; return; }
    grep "Prueba:" $L/train_${exp}_s$s.log
  fi
  local out=metricas/semillas/$exp/s$s
  if [ ! -f $out/resumen.json ]; then
    $PY evaluate_metrics.py --model $modelo --splits test --signers-csv personas_video.csv \
      --seed $s --out-dir $out > $L/metricas_${exp}_s$s.log 2>&1 || echo "FALLO metricas $exp s$s"
  fi
}

cv() {  # experimento semilla tipo epocas [clases...]
  local exp=$1 s=$2 t=$3 e=$4; shift 4
  local out=metricas/semillas_cv/$exp/s$s
  [ -f $out/resumen.json ] && return
  echo "== cv $exp s$s $(date +%H:%M)"
  $PY evaluate_metrics.py --sin-modelo --data-type $t ${1:+--clases "$@"} --signers-csv personas_video.csv \
    --cv 5 --cv-epochs $e --cv-arch ligera --cv-por-persona --cv-incluir-test --cv-solo-originales \
    --seed $s --out-dir $out > $L/cv_${exp}_s$s.log 2>&1 || { echo "FALLO cv $exp s$s"; return; }
  grep "Exactitud media" $L/cv_${exp}_s$s.log
}

for s in $SEMILLAS; do entrenar features2048_ligera $s --data-type features2048; done
for s in $SEMILLAS; do cv cv_features2048 $s features2048 60; done
for s in $SEMILLAS; do entrenar landmarks_ligera $s --data-type landmarks --epochs 300 --patience 50; done
for s in $SEMILLAS; do cv cv_landmarks $s landmarks 100; done
for s in $SEMILLAS; do entrenar probs_ligera $s --data-type probs; done
for s in $SEMILLAS; do
  entrenar disimiles_A_features2048 $s --data-type features2048 --clases $GRUPO_A
  entrenar disimiles_B_features2048 $s --data-type features2048 --clases $GRUPO_B
done
for s in $SEMILLAS; do
  cv cv_disimiles_A_features2048 $s features2048 60 $GRUPO_A
  cv cv_disimiles_B_features2048 $s features2048 60 $GRUPO_B
done
for s in $SEMILLAS; do entrenar features2048_original $s --data-type features2048 --arch original; done
echo "== fin $(date +%H:%M)"
