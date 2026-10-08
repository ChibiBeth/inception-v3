# Documentación técnica — Reconocimiento de señas de la LSPy con MediaPipe, Inception V3 y LSTM

Documentación del código de la tesis (repositorio `inception-v3`, rama `todo_version_ever_con_requirements`). Redactada el 6 de octubre de 2026 a partir de la lectura completa del código y de un relevamiento de `rawdata_clean/`.

## Contenido

| # | Documento | Para qué sirve |
|---|---|---|
| 1 | [Visión general y arquitectura](01_arquitectura.md) | Cómo fluye un video por el sistema, formas de los tensores, capas y parámetros |
| 2 | [Instalación y uso](02_instalacion_y_uso.md) | Entorno, orden de ejecución, argumentos y problemas frecuentes |
| 3 | [Referencia del código](03_referencia_de_codigo.md) | Qué hace cada archivo y cada función |
| 4 | [Conjunto de datos](04_conjunto_de_datos.md) | Cantidades, nombres, personas, propiedades de los videos |
| 5 | [Hallazgos y recomendaciones](05_hallazgos_y_recomendaciones.md) | Problemas encontrados, cómo corregirlos y hoja de ruta |
| 6 | [Métricas para la tesis](06_metricas_para_la_tesis.md) | Uso de `evaluate_metrics.py`, interpretación y cómo llevarlo a LaTeX |
| — | [personas_video.csv](../personas_video.csv) | Persona que signa cada video original de `rawdata/` (`personas_plantilla.csv` es la plantilla vacía) |

## Resumen en una página

**El sistema.** Cada video pasa por MediaPipe Hands, que dibuja el esqueleto de las manos en blanco sobre negro (299 × 299) y guarda sus coordenadas. Cada video se remuestrea a 150 fotogramas equiespaciados, Inception V3 (ImageNet, con su última capa reentrenada opcionalmente) convierte cada dibujo en un vector y una LSTM clasifica la secuencia en una de 10 señas.

**Los datos.** 100 grabaciones reales de 10 personas (`rawdata/`, `personas_video.csv`): 80 en train (7 a 9 por seña, cada una con 4 copias aumentadas: 400 secuencias) y 20 en test (2 por seña, sin copias). La división se hizo antes del aumento. Las personas de test casi no aparecen en train: la prueba mide la generalización a personas nuevas.

**Resultados (7 de octubre de 2026).** 40 % – 45 % de exactitud en prueba según la representación (azar: 10 %; versión anterior: 18 % – 30 %) y 53,5 % ± 6,2 en validación cruzada por persona. Detalle en [06 · 6.8](06_metricas_para_la_tesis.md#68-resultados-de-la-regeneración-completa-7-de-octubre-de-2026).

**Hallazgos de la revisión y estado** (detalle en [05](05_hallazgos_y_recomendaciones.md#estado-de-aplicación-6-de-octubre-de-2026)):

1. **El extractor usaba una capa `Dense(10)` con pesos aleatorios** sobre Inception V3 → corregido: 2048 características por fotograma (`pooling='avg'`). → [H1](05_hallazgos_y_recomendaciones.md#h1)
2. **El `glob` que buscaba los fotogramas mezclaba videos** (103 de 415 secuencias afectadas) → corregido. → [H2](05_hallazgos_y_recomendaciones.md#h2)
3. **El conjunto de prueba se usaba como validación** → ahora la validación sale de train, agrupada por video original, y la prueba se usa una sola vez. → [H3](05_hallazgos_y_recomendaciones.md#h3)
4. **Con 20 videos de prueba, cada error vale 5 puntos.** Hay que informar intervalos de confianza y complementar con validación cruzada (`evaluate_metrics.py`). → [H4](05_hallazgos_y_recomendaciones.md#h4)
5. Alineación con la tesis: el aumento ahora hace rotación, zoom y eliminación de fotogramas; Inception V3 reentrena su última capa y entrega probabilidades por seña; la LSTM por defecto tiene 2 capas. Quedan por corregir en el texto el tamaño del conjunto y “precisión” → “exactitud”. → [tabla de alineación](05_hallazgos_y_recomendaciones.md#puntos-de-la-tesis-a-alinear-con-el-código)

**Para el libro.** `evaluate_metrics.py` genera más de 20 métricas (exactitud con IC95, comparación con el azar, precisión, exhaustividad, F1, kappa, MCC, calibración, ROC, latencia…), tablas LaTeX listas para `\input{}` con coma decimal, figuras en PDF vectorial y una auditoría de los datos.

```bash
python evaluate_metrics.py --model lstm_senha_model --raw-dir rawdata_aug --signers-csv personas_video.csv
```
