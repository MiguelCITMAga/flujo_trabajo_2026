import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_mesh(X, Y, U, folder="", name="", title="", show=False, sol_value=0, error=0, loss=0, iter=0, vmin=None, vmax=None):
    """
    Genera y guarda dos visualizaciones:
    1. Contorno de la función U(x, y)
    2. Superficie 3D de U(x, y)
    
    Parámetros:
    - X, Y: mallas 2D con shape (Nx, Ny)
    - U: función evaluada sobre la malla, shape (Nx, Ny)
    - case: nombre del caso (por estética de zlim)
    - folder: carpeta donde guardar las imágenes
    - name: nombre base de los archivos .png
    - error, loss: métricas opcionales a mostrar en el título
    - iter: número de iteración (para título)
    """

    X, Y, U = X.cpu(), Y.cpu(), U.cpu()

    # Contour plot
    fig, ax = plt.subplots(dpi=200)
    cp = ax.contourf(X, Y, U, 20, cmap="hot", vmin=U.min(), vmax=U.max())
    fig.colorbar(cp)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


    
    if iter: title += f"Iter: {iter}"
    if loss: title += f", Loss: {loss:.5f}"
    if error: title += f"\nRelative error: {error:.5f}"
    if sol_value: title += f", Sol. value (0,0): {sol_value:.5f}"
    ax.set_title(title)

    plt.savefig(f"{folder}/contour_{name}.png")
    if show:
        plt.show()
    plt.close()

    # 3D plot
    # fig = plt.figure(dpi=200)
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(X.numpy(), Y.numpy(), U.numpy(), cmap="rainbow")
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("u(x,y)")
    # ax.set_zlim(-1, 1)


    # plt.savefig(f"{folder}/{name}.png")
    # if show:
    #     plt.show()
    # plt.close()


def plot_training_curves(res_dict, folder):
    """
    Guarda las curvas de entrenamiento (error relativo L2 y pérdida) en escala logarítmica.
    """
    loss_values = res_dict["loss"]
    error_l2_values_mu1 = res_dict["err_l2_mu1"]
    error_l2_values_mu2 = res_dict["err_l2_mu2"]

    # Error relativo L2 para cada parámetro
    plt.figure(dpi=150)
    if len(error_l2_values_mu1) > 0:
        plt.plot(error_l2_values_mu1, label="Error relativo L2 (mu=0.8,0.8)")
    if len(error_l2_values_mu2) > 0:
        plt.plot(error_l2_values_mu2, label="Error relativo L2 (mu=-0.8,-0.8)")
    plt.yscale("log")
    plt.xlabel("Iteración")
    plt.ylabel("Error relativo")
    plt.title("Evolución del error relativo L2 por parámetro")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(folder, "error_relativo_l2.png"))
    plt.close()

    # Función de pérdida
    plt.figure(dpi=150)
    plt.plot(loss_values, label="Pérdida_total", color="orange")
    plt.yscale("symlog")
    plt.xlabel("Iteración")
    plt.ylabel("Pérdida")
    plt.title("Evolución de la función de pérdida")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(folder, "perdida_total.png"))
    plt.close()

