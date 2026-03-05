#################
# Librerías
#################
import logging
import os
import sys
import torch
import numpy as np
import time

from utils.plots import plot_mesh, plot_training_curves
from utils.utils import create_tests_folder
from utils.gen_data import data_gen
from utils.PINN import FNN, ResNet, FNN_comb
from utils.sol_exacta import solve_poisson_fd_two_materials


import torch
import numpy as np
import random





SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#################
#  Configuración del device
#################

# Establecemos el tipo de dato por defecto como número decimal de 32 bits (float32)
torch.set_default_dtype(torch.float)

# Si CUDA está disponible, se usará la GPU; de lo contrario, se usará la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # Usar print temporalmente
if device == "cuda":
    print(torch.cuda.get_device_name())

#Configuración del logging
logging.basicConfig(handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO, format="%(message)s")
logger = logging.getLogger()    

################################################################################################################################
## Problema para resolver:                                                                                                    ##
## -div(k(x,y) grad(u)) = f(x,y) en Ω                                                                                         ##
## u = 0 en ∂Ω                                                                                                                ##
## k(x,y) = k_1 si x<0.5, k_2 si x>=0.5, donde k1 y k2 son parametros que desconocemos. Deben de ser entradas de la red       ##
################################################################################################################################

#################
# Parámetros de la red neuronal
#################

steps = 200000 # Número de iteraciones del optimizador (Adam)
lr = 5e-4  # Tasa de aprendizaje (learning rate)

layers = np.array([7, 64, 64, 64, 64, 1])  # Arquitectura de la red neuronal (4 entradas (2 variables indep, 2 parametros), 4 capas ocultas de 64 neuronas, 1 salida)
layers_x = np.array([2, 64, 64, 64, 5]) # Arquitectura de la red espacial
layers_mu = np.array([5, 64, 64, 64, 5]) # Arquitectura de la red paramétrica

log_freq = 5000  # frecuencia de impresión de metricas y graficas durante el entrenamiento (cada log_freq iteraciones)

## Escogemos la EDP a resolver y el dominio de entrada
physics_domain = [-1, 1, -1, 1] # dominio fisico

# dominio parametrico (mu1_min, mu1_max, mu2_min, mu2_max, beta_min, beta_max, sigma1_min, sigma1_max, sigma2_min, sigma2_max)
parametric_domain = [0.5, 5, 0.5, 5, 0.1, 0.9, -1, 1, -1, 1]  

# Número de datos 
N_test_x = 30  #  Datos test en espacio
N_test_y = 30  #  Datos test en tiempo
N_train = 500 # Numero de puntos de colocacion en espacio (interiores)
N_train_mu = 50 # Numero de puntos de colocacion en espacio parametrico
w_bc= 1 # hiperparametro que acompaña a la perdida en contorno (solo si se usan soft constraints)
w_pde = 1 # hiperparametro que acompaña a la perdida de pde (solo si se usan soft constraints)


# Creamos una carpeta para guardar resultados y configuramos logging
test_folder = create_tests_folder(parent_folder="results")

model_dir = os.path.join(test_folder, "modelos")
os.makedirs(model_dir, exist_ok=True)

if not os.path.exists(test_folder):
    os.makedirs(test_folder)
file_handler = logging.FileHandler(test_folder + "/0_log.txt")
logger.addHandler(file_handler)

#################
# Generate data
#################

DG = data_gen(physics_domain, parametric_domain) 










#################
# Creamos modelo y optimizador
#################

model = ResNet(layers, physics_domain).to(device)
#model = FNN_comb(layers_x, layers_mu).to(device)

logging.info(model)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)

# Dos técnicas distintas de scheduling
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

# Se pueden utilizar otros optimizadores; por ejemplo L-BFGS en la fase de refinamiento
# L-BFGS Optimizer
#optimizer = torch.optim.LBFGS(PINN.parameters(), lr=1e-3, max_iter=500, tolerance_grad=1e-6)



# Creamos un diccionario para guardar distintas métricas a lo largo del entrenamiento
res_dict = {"loss": [], "loss_bc": [], "loss_pde": [], "err_l2_mu1": [], "err_l2_mu2": []}


###
# Resolvemos el problema con diferencias finitas, para dos configuraciones paramétricas dadas. Servirá como validación
###

# FD para k1 = 1, k2 = 4
k_fd1 = np.array([1,4]) 
X_fd1, Y_fd1, u_fd1 = solve_poisson_fd_two_materials(N_test_x, k_fd1[0], k_fd1[1], beta=0.3, sigma = (0.8, 0.8), 
                                                     domain=(physics_domain[0], physics_domain[1])) 
umin1 = u_fd1.min()
umax1 = u_fd1.max()
plot_mesh(torch.tensor(X_fd1), torch.tensor(Y_fd1), torch.tensor(u_fd1), folder=test_folder, name=f"sol_exacta_k1_1_k2_4_bet_03_08_08", title=f"Solución exacta mu=(1, 4, 0.3, 0.8, 0.8)")

# FD para k1 = 3, k2 = 1.5
k_fd2 = np.array([3,1.5])
X_fd2, Y_fd2, u_fd2 = solve_poisson_fd_two_materials(N_test_x, k_fd2[0], k_fd2[1], beta= 0.5, sigma = (-0.8, -0.8), 
                                                     domain=(physics_domain[0], physics_domain[1]))
umin2 = u_fd2.min()
umax2 = u_fd2.max()
plot_mesh(torch.tensor(X_fd2), torch.tensor(Y_fd2), torch.tensor(u_fd2), folder=test_folder, name=f"sol_exacta_k1_3_k2_1.5_bet_05_neg_08_neg_08", title=f"Solución exacta mu=(3, 1.5, 0.5, -0.8, -0.8)")

# Creamos un conjunto de puntos test para evaluar el error a lo largo del entrenamiento, para las dos configuraciones paramétricas de referencia.
# Este conjunto permanece fijo durante todo el entrenamiento.

x_test, X, Y = DG.get_test_dataset(N_test_x, N_test_y)
x_test = x_test.to(device)
mu_1 = torch.tensor([[1.0, 4.0, 0.3, 0.8, 0.8]], dtype=torch.float32, device=device)
mu_1_exp = mu_1.expand(x_test.shape[0], -1)
inputs_1_base = torch.cat([x_test, mu_1_exp], dim=1)
mu_2 = torch.tensor([[3.0, 1.5, 0.5, -0.8, -0.8]], dtype=torch.float32, device=device)
mu_2_exp = mu_2.expand(x_test.shape[0], -1)
inputs_2_base = torch.cat([x_test, mu_2_exp], dim=1)

start_time_a = time.time()
count = 1
logging.info("Starting training...")

###
# Bucle de optimización de la red neuronal
# En cada iteración se generan nuevos puntos para entrenar, buscando un cubrimiento eficiente tanto del dominio físico como del paramétrico.
###

# beta = 0.99
# loss_emma = 0
# tol = 1e-10
for i in range(1, steps + 1):
    model.train() # modelo en modo entrenamiento
    
    
    #pts_pde, pts_param, _ = DG.get_train_points(N_train, 2, N_train_mu, 5, device = device) # no queremos datos en el contorno, porque estamos usando hard constraints
    _, _, pts_train = DG.get_train_points(N_train, 2, N_train_mu, 5, device = device) # obtenemos un conjunto de puntos que incluye tanto interiores como frontera
    
    # Evaluamos la pérdida. Usamos loss_pde si queremos hard constraints, asegurarnos de en PINN.py descomentar las líneas que multiplican la red por eta

    
    loss = model.loss_pde(pts_train, domain_volume=4.0)
    #loss = model.loss(pts_pde, pts_param)

       
    optimizer.zero_grad() # limpiamos gradientes acumulados
    loss.backward() # backpropagation para calcular gradientes
    optimizer.step() # actualizamos pesos de la red usando el optimizador (Adam)

    # ReduceLROnPlateau espera un argumento de entrada, en este caso loss. Si se usa StepLR daría error, no ponemos argumento. 
    scheduler.step()  # Actualiza la tasa de aprendizaje usando la pérdida actual

    #Guardamos pérdida en el diccionario de resultados
    res_dict["loss"].append(loss.item())

    # Calculamos error L2 usando el conjunto de puntos test
    model.eval()
    with torch.no_grad():
        u_pred_1 = model(inputs_1_base).squeeze().reshape(N_test_x, N_test_y)
        u_pred_2 = model(inputs_2_base).squeeze().reshape(N_test_x, N_test_y)
        # u_pred_1 = model(x_test, mu_1_exp).squeeze().reshape(N_test_x, N_test_y)
        # u_pred_2 = model(x_test, mu_2_exp).squeeze().reshape(N_test_x, N_test_y)
        # Normas L2
        err_l2_1 = torch.norm(u_pred_1 - torch.tensor(u_fd1, device=device), p=2) / torch.norm(torch.tensor(u_fd1, device=device), p=2) 
        err_l2_2 = torch.norm(u_pred_2 - torch.tensor(u_fd2, device=device), p=2) / torch.norm(torch.tensor(u_fd2, device=device), p=2)
        # Guardar métricas por separado
        res_dict["err_l2_mu1"].append(err_l2_1.item())
        res_dict["err_l2_mu2"].append(err_l2_2.item())

        
    if i % log_freq == 0 or i == 1:
        model.eval() # modelo en modo evaluación
        current_lr = optimizer.param_groups[0]['lr'] # tasa de aprendizaje actual para mostrarla en el log
        logger.info(f"| Iter: {i} | Loss: {loss.item():.4f} |Error L2 mu1: {err_l2_1.item():.4f} |Error L2 mu2: {err_l2_2.item():.4f} |lr: {current_lr:.2e} | Total_time: {(time.time() - start_time_a)/60:.1f}min")
        
        # Gráficas para parámetros específicos
        x_test, X, Y = DG.get_test_dataset(N_test_x, N_test_y)
        x_test = x_test.to(device)
        
        # Parámetros k1 = 1, k2 = 4
        mu_1 = torch.tensor([[1.0, 4.0, 0.3, 0.8, 0.8]], dtype=torch.float32, device=device)
        mu_1_exp = mu_1.expand(x_test.shape[0], -1)
        inputs_1 = torch.cat([x_test, mu_1_exp], dim=1)
        with torch.no_grad():
            u_pred_1 = model(inputs_1).squeeze()
            u_pred_1 = u_pred_1.reshape(N_test_x, N_test_y)
            #u_pred_1 = model(x_test, mu_1_exp).squeeze().reshape(N_test_x, N_test_y)
        plot_mesh(X, Y, u_pred_1, folder=test_folder, name=f"iter_{i}_k_1_4_03_08_08", title=f"Solución aproximada k=(1, 4, 0.3, 0.8, 0.8)",iter=i, loss=loss.item(), error=err_l2_1.item())
        
        error_1 = torch.abs(u_pred_1 - torch.tensor(u_fd1, device=device))
        plot_mesh(X, Y, error_1, folder=test_folder, name=f"error_iter_{i}_k_1_4_03_08_08", title=f"Error de aproximación k=(1, 4, 0.3, 0.8, 0.8)", iter=i, loss=loss.item())
        
        # Parámetros k1 = 3, k2 = 1.5
        mu_2 = torch.tensor([[3.0, 1.5, 0.5, -0.8, -0.8]], dtype=torch.float32, device=device)
        mu_2_exp = mu_2.expand(x_test.shape[0], -1)
        inputs_2 = torch.cat([x_test, mu_2_exp], dim=1)
        with torch.no_grad():
            u_pred_2 = model(inputs_2).squeeze()
            u_pred_2 = u_pred_2.reshape(N_test_x, N_test_y)
            #u_pred_2 = model(x_test, mu_2_exp).squeeze().reshape(N_test_x, N_test_y)
        plot_mesh(X, Y, u_pred_2, folder=test_folder, name=f"iter_{i}_k_3_1.5_05_neg_08_neg_08", title=f"Solución aproximada k=(3, 1.5, 0.5, -0.8, -0.8)", iter=i, loss=loss.item(), error=err_l2_2.item())
        
        error_2 = torch.abs(u_pred_2 - torch.tensor(u_fd2, device=device))
        plot_mesh(X, Y, error_2, folder=test_folder, name=f"error_iter_{i}_k_3_1.5_05_neg_08_neg_08", title=f"Error de aproximación k=(3, 1.5, 0.5, -0.8, -0.8)", iter=i, loss=loss.item())

    # loss_emma = 1/(1-beta**i) * (beta * loss_emma + (1 - beta) * loss.item())
    # if loss_emma < 0 and np.abs(loss.item() - loss_emma)/loss.item() < tol:
    #     logger.info(f"Learning concludes at iteration {i} with loss {loss.item():.4f}")
    #     break
        
        
        
logging.info('Finished training')
torch.save(model.state_dict(), os.path.join(model_dir, "trained_model.pth"))

# --- AÑADIR: fase final con L-BFGS (refinamiento) ---
# logging.info(f"---------------------------------------------------------------------------")
# logging.info("Starting L-BFGS refinement...")

# # 1. L-BFGS necesita un conjunto fijo de datos para evaluar el gradiente consistentemente
# N_total_x_lbfgs = 1000 # Un tamaño que entre bien en la GPU
# N_total_mu_lbfgs = 200
# _, _, pts_train_lbfgs = DG.get_train_points(N_total_x_lbfgs, 2, N_total_mu_lbfgs, 5, device=device)

# # Concatenamos las 2 espaciales + 5 paramétricas (como espera FNN)
# pts_train_lbfgs = pts_train_lbfgs.clone().detach().requires_grad_(True)

# # 2. Configurar el optimizador L-BFGS
# lbfgs_opt = torch.optim.LBFGS(model.parameters(), 
#                               max_iter=20, # Número de refinamientos internos por paso
#                               tolerance_grad=1e-7, 
#                               tolerance_change=1e-9, 
#                               history_size=50, 
#                               line_search_fn='strong_wolfe')

# # 3. La función closure que L-BFGS evalúa múltiples veces
# def closure():
#     lbfgs_opt.zero_grad()
#     loss = model.loss_pde(pts_train_lbfgs, domain_volume=4.0)
#     loss.backward()
#     return loss

# # 4. Bucle externo de L-BFGS
# for i in range(1, 301):
#     loss_val = lbfgs_opt.step(closure)
    
#     if i % 10 == 0 or i == 1:
#         logger.info(f"L-BFGS | Step: {i}/300 | Loss: {loss_val.item():.5e}")

# final_loss_pde = model.loss_pde(pts_train_lbfgs).item()
# logging.info(f"L-BFGS finished. Final loss_pde={final_loss_pde:.5e}")


# # Guardamos nuestro modelo entrenado para evaluaciones futuras
# torch.save(model.state_dict(), os.path.join(model_dir, "trained_net_lbfgs.pth"))

model.eval()
with torch.no_grad():
    # Comparación para k1 = 1, k2 = 4
    mu_1 = torch.tensor([[1.0, 4.0, 0.3, 0.8, 0.8]], dtype=torch.float32, device=device)
    mu_1_exp = mu_1.expand(x_test.shape[0], -1)
    inputs_1 = torch.cat([x_test, mu_1_exp], dim=1)
    u_pred_1 = model(inputs_1).squeeze().reshape(N_test_x, N_test_y).cpu()
    #u_pred_1 = model(x_test, mu_1_exp).squeeze().reshape(N_test_x, N_test_y).cpu()
    error_l2_1 = torch.norm(u_pred_1 - torch.tensor(u_fd1), p=2) / torch.norm(torch.tensor(u_fd1), p=2)
    logger.info(f"Relative L2 error for k=(1, 4, 0.3, 0.8, 0.8): {error_l2_1:.5f}")

    # Comparación para k1 = 3, k2 = 1.5
    mu_2 = torch.tensor([[3.0, 1.5, 0.5, -0.8, -0.8]], dtype=torch.float32, device=device)
    mu_2_exp = mu_2.expand(x_test.shape[0], -1)
    inputs_2 = torch.cat([x_test, mu_2_exp], dim=1)
    u_pred_2 = model(inputs_2).squeeze().reshape(N_test_x, N_test_y).cpu()
    #u_pred_2 = model(x_test, mu_2_exp).squeeze().reshape(N_test_x, N_test_y).cpu()
    error_l2_2 = torch.norm(u_pred_2 - torch.tensor(u_fd2), p=2) / torch.norm(torch.tensor(u_fd2), p=2)
    logger.info(f"Relative L2 error for k=(3, 1.5, 0.5, -0.8, -0.8): {error_l2_2:.5f}")

# Graficamos curvas de entrenamiento (loss y l2). Con Deep Ritz no son demasiado explicativas, pero pueden ayudar a detectar problemas de convergencia o sobreajuste.
plot_training_curves(res_dict, test_folder)
