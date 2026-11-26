# Prueba de unos minutos para ver que todo funciona en este equipo antes de lanzar correr.py.
# Entrena 2 epocas con pocos datos (una corrida de cada tipo) y deja todo en resultados_prueba/.
# %%
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
import torch
import config as cfg
import entrenar as E

preset_real, raiz_real = cfg.PRESET, cfg.RAIZ
eq = cfg.info_equipo()
print(f"{eq['torch']} | {eq['dispositivo']} | {eq['gpu']} | {eq['sistema']}")
if cfg.DEVICE != "cuda":
    print("OJO: PyTorch no ve una GPU. Revisar la instalacion de PyTorch con CUDA antes de lanzar correr.py.")

# tiempo estimado de la cola real
COLA_REAL = E.cola_experimentos()
s_sam, s_sgd, horas = E.estimar_horas(COLA_REAL, pasos=30 if cfg.DEVICE == "cuda" else 2)
print(f"cola real ({preset_real}): {len(COLA_REAL)} corridas | SAM {s_sam:.0f} s/época | SGD {s_sgd:.0f} s/época | aprox. {horas:.1f} h en total")

# %% corridas cortas
cfg.PRESET = "humo"
cfg.RAIZ = os.path.join(cfg.CARPETA, "resultados_prueba")
vistos, cola = set(), []
for c in E.cola_experimentos():
    k = (c["metodo"], c["lam"] > 0, c["momentum"] > 0)
    if k not in vistos:
        vistos.add(k)
        cola.append(c)
t0 = time.time()
for c in cola:
    E.entrenar(c)
E.escribir_estado(cola)
E.escribir_resumen()
print(f"\nprueba terminada en {(time.time() - t0) / 60:.1f} min: {len(cola)} corridas cortas sin errores")
print("si la precisión de prueba queda cerca de 10% es normal: son 2 épocas con pocos datos")
cfg.PRESET, cfg.RAIZ = preset_real, raiz_real
