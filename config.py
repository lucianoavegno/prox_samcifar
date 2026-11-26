# Configuracion. Es el unico archivo que hay que tocar.
import os, platform
import torch

PRESET   = "minimo"                        # "minimo": 60 epocas y 1 semilla; "completo": 200 epocas y 3 semillas (lo del paper)
GRUPOS   = ["L", "R", "P", "N1", "S"]      # opcionales: "N2", "N3"
DATASETS = ["cifar10"]                     # agregar "cifar100" si alcanza el tiempo

PRESETS = {
    "humo":     dict(epocas=2,   semillas=[0],       sub_train=2048, sub_test=1000),   # lo usa prueba_rapida.py
    "minimo":   dict(epocas=60,  semillas=[0],       sub_train=None, sub_test=None),
    "completo": dict(epocas=200, semillas=[0, 1, 2], sub_train=None, sub_test=None),
}

BASE = dict(
    modelo="resnet18", batch=128, wd=1e-3, eta1=0.1, rho0=0.05,
    momentum_R=0.9,                  # momentum de la replica del paper (grupo R)
    momentum=0.0,                    # momentum de los demas grupos
    eta_const=0.01,                  # paso constante de los grupos con lambda > 0
    val_frac=0.1, cutout=16, cutout_p=0.5, delta=0.01,
    lam_ref=1e-4,                    # lambda de los grupos P, N1 y S (elegirlo con el piloto)
    lam_piloto=[1e-5, 1e-4, 1e-3],   # valores de lambda del piloto (grupo L)
    epocas_piloto=10,
    lam_grid=[5e-5, 2e-4],           # valores de lambda para N2
    tbar=0.1,                        # t barra del certificado, el mismo en todas las corridas
    nu_rec=0.5,                      # nu de Prox-SAM recortado
    L0=100.0,                        # valor inicial de la estimacion de L
    rho_usam=0.005,                  # radio de Prox-USAM para N3 (se necesita L*rho < 1)
    # que corridas entran en cada grupo
    reglas_R=["const", "dim1", "dim2", "dim3"],      # reglas de paso de la replica
    etas_R=[0.01], etas_R_completo=[0.1, 0.01, 0.001],   # pasos constantes de la replica
    reglas_P=["const", "dim1", "dim2", "dim3"],      # reglas de paso de Prox-SAM con radio constante
    reglas_P_sgd=["const", "dim2"],                  # reglas de paso de Prox-SGD
    radios_N1=["hip31", "paper9"],                   # radios decrecientes
    # certificado
    cada_certificado=5,              # cada cuantas epocas se calcula
    n_certificado=5000,              # imagenes usadas
    # guardado y avance
    min_guardado=5,                  # minutos entre guardados dentro de una epoca
    cada_print=50,                   # cada cuantas iteraciones se imprime el avance
    max_horas=None,                  # si es un numero, para solo (guardando) despues de esas horas
    estimar_tiempo=True,             # medir el tiempo por epoca antes de partir
)

# carpetas: todo queda al lado de este archivo
CARPETA = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.environ.get("PROXSAM_RESULTADOS", os.path.join(CARPETA, "resultados"))
DATOS = os.path.join(CARPETA, "datos")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


def P():
    return PRESETS[PRESET]


def info_equipo():
    gpu = "sin GPU"
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        gpu = f"{p.name}, {p.total_memory / 2**30:.1f} GB"
    return dict(torch=f"PyTorch {torch.__version__}", dispositivo=DEVICE, gpu=gpu,
                sistema=f"{platform.system()} {platform.release()}", python=platform.python_version())
