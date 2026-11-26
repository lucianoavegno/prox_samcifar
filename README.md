# Prox-SAM en CIFAR: experimento completo

ResNet-18 en CIFAR-10 con los mismos parámetros del notebook `prox_sam_cifar_colab.ipynb`: 60 épocas, una semilla, 16 corridas (810 épocas en total). Con `PRESET = "completo"` en `config.py` pasa a 200 épocas y 3 semillas, que es lo que usa el paper.

La carpeta `cifar_reducido/` tiene el mismo código con un experimento más corto.

## Qué hay que tener instalado

Python con PyTorch (con CUDA, para que use la GPU), torchvision, numpy, pandas, matplotlib y scipy. CIFAR-10 se descarga solo la primera vez (170 MB) a la carpeta `datos/`.

## Cómo correrlo en Spyder

1. Abrir `prueba_rapida.py` y ejecutarlo con F5. Toma unos minutos. Dice qué GPU ve PyTorch, cuántas horas tomaría todo en ese equipo, y entrena un par de épocas con pocos datos para comprobar que no hay errores. Si dice «PyTorch no ve una GPU», hay que arreglar eso antes de seguir: en CPU no es viable.
2. Abrir `correr.py` y ejecutarlo con F5. Lanza todas las corridas, una tras otra.
3. Cuando termine (o en cualquier momento, con resultados parciales), ejecutar `analisis.py` para las tablas y figuras.
4. Opcional: `extras_hessiano_poda.py` calcula el autovalor máximo del hessiano y la robustez a la poda de cada corrida terminada.

Todos los parámetros están en `config.py`; los demás archivos no hay que tocarlos.

## Guardado y cómo retomar

- Cada corrida guarda su avance al final de cada época y, dentro de la época, cada 5 minutos (`min_guardado`).
- Si el equipo se apaga, se corta la luz o se cierra Spyder, basta volver a ejecutar `correr.py`: salta las corridas terminadas y retoma la que estaba a medias desde el último guardado.
- Para parar sin perder nada: crear un archivo vacío llamado `PARAR.txt` dentro de `resultados/`. El programa guarda y se detiene en unos segundos. También se puede poner un límite de horas en `max_horas`.
- Con el botón de stop de Spyder se pierde, a lo más, lo hecho desde el último guardado.
- Mientras corre, imprime una línea cada 50 iteraciones con la pérdida, los segundos por iteración y cuánto falta de la época.

## Qué queda guardado

En `resultados/`:

| Archivo | Contenido |
|---|---|
| `estado.csv` | una línea por corrida: terminada, a medias o pendiente, y épocas hechas |
| `resumen.csv` | una línea por corrida terminada con la configuración y las métricas finales |
| `registro.txt` | copia de lo que se imprimió, con fecha y hora |
| `figuras/` | las figuras de `analisis.py`, en PDF y PNG |

En la carpeta de cada corrida:

| Archivo | Contenido |
|---|---|
| `config.json` | parámetros de la corrida y datos del equipo (GPU, versión de PyTorch) |
| `log.csv` | una fila por época: paso, radio, pérdida, precisión de entrenamiento, validación y prueba, fracción de pesos no nulos, los dos certificados, norma del gradiente, radio efectivo, estimación de L y segundos de la época |
| `iter_log.csv` | una fila cada 50 iteraciones: pérdida del lote y segundos por iteración |
| `final.json` | métricas finales, densidad por capa y horas totales |
| `ckpt.pt` | pesos de la red y estado del optimizador (sirve para retomar y para `extras_hessiano_poda.py`) |
| `extras.json` | hessiano y poda, si se corrió el script opcional |

Los certificados (`certificado` es la norma del mapeo de gradiente proximal y `dist_fuerte` la distancia al subdiferencial) se calculan en la primera época, cada 5 épocas y en la última.

## Los grupos de corridas

| Grupo | Qué corre | Para qué |
|---|---|---|
| L | Prox-SGD, pocas épocas, varios λ | elegir λ (`lam_ref`) |
| R | SAM con λ = 0 y momentum 0,9, reglas de paso del paper | réplica de la Tabla 4 de Khanh, Luong, Mordukhovich y Tran |
| P | Prox-SAM con radio constante y λ > 0, y Prox-SGD | piso del radio constante en el problema compuesto |
| N1 | radios decrecientes con λ > 0 | comparar con el radio constante |
| S | Prox-SAM recortado | método sin piso |

Los grupos N2 (varios λ) y N3 (Prox-USAM) están apagados; se activan agregándolos a `GRUPOS`.

`lam_ref` está en 1e-4 sin ajustar. Lo ideal es mirar la tabla del piloto (primera tabla de `analisis.py`, sale apenas terminan las corridas L) y, si con 1e-4 casi no hay pesos en cero, cambiarlo en `config.py` y volver a ejecutar `correr.py`. Las corridas del grupo R no dependen de λ y no se repiten.

## Para Colab

Subir la carpeta, elegir GPU y ejecutar `!python correr.py` desde ella. Para que los resultados queden en Drive, definir antes la variable de entorno `PROXSAM_RESULTADOS` con la ruta de la carpeta de Drive.
