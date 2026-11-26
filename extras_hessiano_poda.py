# Opcional, despues de entrenar: autovalor maximo del hessiano y robustez a la poda de cada corrida terminada.
# Lee los checkpoints y escribe extras.json en la carpeta de cada corrida.
# %%
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
import torch
import config as cfg
from datos import datos_para
from modelo import crear_modelo
from metricas import autovalor_hessiano, robustez_poda

for f_fin in sorted(glob.glob(os.path.join(cfg.RAIZ, "*", "final.json"))):
    carpeta = os.path.dirname(f_fin)
    f_ext = os.path.join(carpeta, "extras.json")
    if os.path.exists(f_ext):
        continue
    c = json.load(open(f_fin))["config"]
    d = datos_para(c["dataset"])
    modelo = crear_modelo(c["modelo"], d["n_clases"])
    st = torch.load(os.path.join(carpeta, "ckpt.pt"), map_location=cfg.DEVICE, weights_only=False)
    modelo.load_state_dict(st["modelo"])
    Xc = d["Xtr"][: cfg.BASE["n_certificado"], :, 4:36, 4:36]
    Yc = d["Ytr"][: cfg.BASE["n_certificado"]]
    ext = {}
    try:
        ext["lambda_max_hessiano"] = autovalor_hessiano(modelo, Xc, Yc, d)
    except Exception as e:
        print("  (hessiano omitido:", e, ")")
    ext["robustez_poda"] = robustez_poda(modelo, d)
    json.dump(ext, open(f_ext, "w"), indent=1)
    print(os.path.basename(carpeta), "->", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ext.items()}, flush=True)
