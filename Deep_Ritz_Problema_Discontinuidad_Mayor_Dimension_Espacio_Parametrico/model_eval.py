"""
============================================================
EVALUACIÓN DEL MODELO PINN PARAMÉTRICO (5 PARÁMETROS)
============================================================
Este script evalúa un modelo PINN FNN entrenado para el problema
de Poisson con conductividad discontinua móvil y fuente móvil.
Calcula la solución aproximada, la solución exacta por diferencias 
finitas, los errores y guarda los gráficos de contorno.
Incluye análisis estadístico sobre N muestras aleatorias paramétricas.
============================================================
"""

import os
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# Utils del proyecto
from utils.PINN import FNN
from utils.sol_exacta import solve_poisson_fd_two_materials
from utils.gen_data import data_gen
from utils.plots import plot_mesh


# =========================================================
# 1. CONFIGURACIÓN DEL MODELO Y DIRECTORIOS
# =========================================================

# ⚠ IMPORTANTE: Actualiza este path con la ruta real al archivo .pth de tu último entrenamiento
MODEL_PATH = r"C:\Users\Miguel\Repositorios\Proyecto_2026\Repsol\Parametric_Solvers\Deep_Ritz_Problema_Discontinuidad_Mayor_Dimension_Espacio_Parametrico\results\test_20260304-111825\modelos\trained_net.pth"

RESULTS_FOLDER = "./results/eval"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Parámetros del dominio físico y paramétrico
DOMAIN = [-1, 1, -1, 1]
# (k1_min, k1_max, k2_min, k2_max, beta_min, beta_max, sigma1_min, sigma1_max, sigma2_min, sigma2_max)
PARAMETRIC_DOMAIN = [0.5, 5, 0.5, 5, 0.1, 0.9, -1, 1, -1, 1]

# Resoluciones de test
N_TEST_X = 30
N_TEST_Y = 30

# Lista de parámetros [k1, k2, beta, sigma1, sigma2] a evaluar visualmente
PARAM_LIST = [
    [1.0, 4.0, 0.3, 0.8, 0.8],     # Caso 1 clásico
    [3.0, 1.5, 0.5, -0.8, -0.8],   # Caso 2 clásico
    [5.0, 0.5, 0.2, 0.0, 0.0]      # Caso de alto contraste con fuente en el centro
]

# Configuración del Análisis Estadístico
N_SAMPLES = 100  # Número de configuraciones aleatorias de 5 parámetros a evaluar

# Arquitectura del modelo FNN (2 espaciales + 5 parámetros = 7 entradas)
LAYERS = np.array([7, 128, 128, 128, 1])

# Dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. CARGA DEL MODELO
# =========================================================
def load_model(model_path, domain):
    """
    Carga el modelo PINN clásico desde disco.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo de modelo:\n{model_path}")

    model = FNN(LAYERS, domain).to(DEVICE)
    print(model)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✔ Modelo cargado desde: {model_path}")
    return model


# =========================================================
# 3. EVALUACIÓN PARA UN SET DE PARÁMETROS
# =========================================================
def evaluate_for_params(model, x_test, X, Y, param_set, n_test_x, n_test_y):
    """
    Ejecuta la predicción y el cálculo de errores para un set de 5 parámetros concreto.
    """
    k1, k2, beta, sigma1, sigma2 = param_set
    
    # 1. Preparar tensor de entrada para la red neuronal
    param_tensor = torch.tensor([param_set], dtype=torch.float32, device=DEVICE)
    param_expanded = param_tensor.expand(x_test.shape[0], -1)
    inputs = torch.cat([x_test, param_expanded], dim=1)

    # 2. Predicción de la PINN
    with torch.no_grad():
        u_pred = model(inputs).squeeze().reshape(n_test_x, n_test_y).cpu().numpy()

    # 3. Solución exacta (diferencias finitas)
    X_fd, Y_fd, u_fd = solve_poisson_fd_two_materials(
        n_test_x, 
        k1, 
        k2, 
        beta=beta,
        sigma=(sigma1, sigma2), 
        domain=(DOMAIN[0], DOMAIN[1])
    )

    # 4. Cálculo de Errores
    err_abs = np.abs(u_pred - u_fd)
    err_l2 = np.linalg.norm(u_pred - u_fd) / np.linalg.norm(u_fd)
    err_max = np.max(err_abs)

    print(f"\nParámetros: k=({k1}, {k2}), beta={beta}, sigma=({sigma1}, {sigma2})")
    print(f"  • Error relativo L2    : {err_l2:.5e}")
    print(f"  • Error absoluto máximo: {err_max:.5e}")
    
    u_min = u_fd.min()
    u_max = u_fd.max()
    
    # Format name string safely
    name_str = f"k_{k1}_{k2}_b_{beta}_s_{sigma1}_{sigma2}".replace(".", "")
    
    # 5. Generación de Gráficas
    plot_mesh(torch.tensor(X), torch.tensor(Y), torch.tensor(u_pred),
              folder=RESULTS_FOLDER, name=f"pred_{name_str}", 
              title=f"Pred: k=({k1},{k2}) b={beta} s=({sigma1},{sigma2})", iter=0, loss=0.0, vmin=u_min, vmax=u_max)

    plot_mesh(torch.tensor(X_fd), torch.tensor(Y_fd), torch.tensor(u_fd),
              folder=RESULTS_FOLDER, name=f"exacta_{name_str}", 
              title=f"Exacta: k=({k1},{k2}) b={beta} s=({sigma1},{sigma2})", iter=0, loss=0.0, vmin=u_min, vmax=u_max)

    plot_mesh(torch.tensor(X), torch.tensor(Y), torch.tensor(err_abs),
              folder=RESULTS_FOLDER, name=f"error_{name_str}", 
              title=f"Error Abs: k=({k1},{k2}) b={beta} s=({sigma1},{sigma2})", iter=0, loss=0.0)


# =========================================================
# 4. ANÁLISIS ESTADÍSTICO
# =========================================================
def run_statistical_analysis(model, x_test, n_test_x, n_test_y):
    """
    Evalúa el modelo sobre N_SAMPLES configuraciones aleatorias (5D),
    calcula métricas estadísticas de error y guarda un histograma.
    """
    print("\n======================================================")
    print(f"   INICIANDO ANÁLISIS ESTADÍSTICO ({N_SAMPLES} MUESTRAS)")
    print("======================================================")

    # Definir límites de muestreo para los 5 parámetros
    mins = np.array([PARAMETRIC_DOMAIN[0], PARAMETRIC_DOMAIN[2], PARAMETRIC_DOMAIN[4], PARAMETRIC_DOMAIN[6], PARAMETRIC_DOMAIN[8]])
    maxs = np.array([PARAMETRIC_DOMAIN[1], PARAMETRIC_DOMAIN[3], PARAMETRIC_DOMAIN[5], PARAMETRIC_DOMAIN[7], PARAMETRIC_DOMAIN[9]])

    # Generar muestras aleatorias uniformes en 5D
    samples = np.random.uniform(low=mins, high=maxs, size=(N_SAMPLES, 5))

    errors = []
    t0 = time.time()

    for i, param_set in enumerate(samples):
        k1, k2, beta, sigma1, sigma2 = param_set

        # 1. Preparar tensores
        param_tensor = torch.tensor([param_set], device=DEVICE, dtype=torch.float32)
        param_expanded = param_tensor.expand(x_test.shape[0], -1)
        inputs = torch.cat([x_test, param_expanded], dim=1)
        
        # 2. Predicción
        with torch.no_grad():
            u_pred = model(inputs).squeeze().reshape(n_test_x, n_test_y).cpu().numpy()

        # 3. Solución exacta
        _, _, u_fd = solve_poisson_fd_two_materials(
            n_test_x, k1, k2, beta=beta, sigma=(sigma1, sigma2), domain=(DOMAIN[0], DOMAIN[1])
        )

        # 4. Cálculo de Error
        err_l2 = np.linalg.norm(u_pred - u_fd) / np.linalg.norm(u_fd)
        errors.append(err_l2)

        if (i + 1) % 20 == 0:
            print(f"  Procesado {i+1}/{N_SAMPLES} muestras...")

    t1 = time.time()

    # Calcular Estadísticas
    errors = np.array(errors)
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    std_err = np.std(errors)
    min_err = np.min(errors)
    max_err = np.max(errors)
    p95_err = np.percentile(errors, 95)

    print("\n--- RESULTADOS ESTADÍSTICOS ---")
    print(f"  • Media (Mean):           {mean_err:.5e}")
    print(f"  • Mediana (Median):       {median_err:.5e}")
    print(f"  • Desv. Est. (Std):       {std_err:.5e}")
    print(f"  • Mínimo (Min):           {min_err:.5e}")
    print(f"  • Máximo (Max):           {max_err:.5e}")
    print(f"  • Percentil 95:           {p95_err:.5e}")
    print(f"  • Tiempo de evaluación:   {t1 - t0:.2f} s")

    # Guardar Histograma
    plt.figure(figsize=(8, 6), dpi=150)
    plt.hist(errors, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean_err, color='red', linestyle='dashed', linewidth=2, label=f'Media: {mean_err:.4f}')
    plt.axvline(median_err, color='orange', linestyle='dashed', linewidth=2, label=f'Mediana: {median_err:.4f}')
    
    plt.title(f'Distribución del Error Relativo L2 ({N_SAMPLES} muestras en 5 parámetros)')
    plt.xlabel('Error Relativo L2')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(alpha=0.3)
    
    hist_path = os.path.join(RESULTS_FOLDER, "histograma_errores_L2.png")
    plt.savefig(hist_path)
    plt.close()
    
    print(f"\n✅ Histograma guardado en: {hist_path}")


# =========================================================
# 5. PROCESO PRINCIPAL DE EVALUACIÓN
# =========================================================
def main():
    print("======================================================")
    print("   EVALUACIÓN DEL MODELO PINN (5 PARÁMETROS)")
    print("======================================================")

    # Cargar modelo
    model = load_model(MODEL_PATH, DOMAIN)

    # Generación de dataset de test
    DG = data_gen(DOMAIN, PARAMETRIC_DOMAIN)
    x_test, X, Y = DG.get_test_dataset(N_TEST_X, N_TEST_Y)
    x_test = x_test.to(DEVICE)

    # Evaluar visualmente cada configuración en PARAM_LIST
    for param_set in PARAM_LIST:
        evaluate_for_params(model, x_test, X, Y, param_set, N_TEST_X, N_TEST_Y)
        
    # Ejecutar el análisis estadístico global
    run_statistical_analysis(model, x_test, N_TEST_X, N_TEST_Y)
        
    print("\n✅ Evaluación general completada. Gráficos guardados en:", RESULTS_FOLDER)

# =========================================================
# 6. EJECUCIÓN
# =========================================================
if __name__ == "__main__":
    main()