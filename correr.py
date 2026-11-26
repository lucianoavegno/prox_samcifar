# Lanza la cola de experimentos. En Spyder: abrir este archivo y apretar F5 (o correr celda por celda).
# Se puede cortar cuando sea y volver a ejecutar: retoma donde quedo.
# %% preparacion
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
import torch
import config as cfg
from entrenar import (cola_experimentos, id_corrida, entrenar, estado_corrida, escribir_estado, escribir_resumen,
                      estimar_horas, registrar, Control, Detener)

os.makedirs(cfg.RAIZ, exist_ok=True)
eq = cfg.info_equipo()
registrar(f"=== inicio | preset {cfg.PRESET} | grupos {cfg.GRUPOS} | datasets {cfg.DATASETS}")
registrar(f"    {eq['torch']} | {eq['dispositivo']} | {eq['gpu']} | {eq['sistema']}")
if cfg.DEVICE != "cuda":
    registrar("    OJO: PyTorch no ve una GPU. En CPU cada epoca toma del orden de 10 a 30 minutos.")
registrar(f"    resultados en: {cfg.RAIZ}")

# %% cola y tiempo estimado
COLA = cola_experimentos()
escribir_estado(COLA)
for c in COLA:
    est, ep, _ = estado_corrida(c)
    print(f"   [{c['grupo']:2s}] {est:10s} {ep:3d}/{c['epocas']:<3d} {id_corrida(c)}")
if cfg.BASE["estimar_tiempo"]:
    s_sam, s_sgd, horas = estimar_horas(COLA, pasos=30 if cfg.DEVICE == "cuda" else 2)
    registrar(f"    SAM: {s_sam:.0f} s/época | SGD: {s_sgd:.0f} s/época | faltan aprox. {horas:.1f} h")
    if cfg.DEVICE == "cuda":
        registrar(f"    memoria de GPU usada: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")

# %% ejecutar
# para parar sin perder nada: crear un archivo PARAR.txt en la carpeta de resultados,
# o fijar BASE["max_horas"] en config.py. Con el boton de stop de Spyder se pierde a lo mas
# lo hecho desde el ultimo guardado (BASE["min_guardado"] minutos).
ruta_parar = os.path.join(cfg.RAIZ, "PARAR.txt")
if os.path.exists(ruta_parar):
    os.remove(ruta_parar)
control = Control(max_horas=cfg.BASE["max_horas"])
try:
    for i, c in enumerate(COLA):
        registrar(f"=== [{i + 1}/{len(COLA)}] grupo {c['grupo']} ({(time.time() - control.t0) / 3600:.2f} h transcurridas)")
        entrenar(c, control)
        escribir_estado(COLA)
        escribir_resumen()
    registrar(f"=== cola terminada en {(time.time() - control.t0) / 3600:.2f} h")
except Detener as e:
    registrar(f"=== detenido: {e}. Para seguir, volver a ejecutar correr.py")
except KeyboardInterrupt:
    registrar("=== interrumpido a mano. Para seguir, volver a ejecutar correr.py (retoma desde el ultimo guardado)")
finally:
    escribir_estado(COLA)
    escribir_resumen()
