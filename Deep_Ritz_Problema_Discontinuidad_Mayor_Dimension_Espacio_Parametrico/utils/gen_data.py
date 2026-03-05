import torch
import numpy as np
from scipy.stats.qmc import Halton

class data_gen:
    def __init__(self, domain, domain_parametric,case=None):
        super().__init__()
        self.case = case
        self.x_min = domain[0]
        self.x_max = domain[1]
        self.y_min = domain[2]
        self.y_max = domain[3]
        self.mu1_min = domain_parametric[0]
        self.mu1_max = domain_parametric[1]
        self.mu2_min = domain_parametric[2]
        self.mu2_max = domain_parametric[3]
        self.beta_min = domain_parametric[4]
        self.beta_max = domain_parametric[5]
        self.sigma1_min = domain_parametric[6]
        self.sigma1_max = domain_parametric[7]
        self.sigma2_min = domain_parametric[8]
        self.sigma2_max = domain_parametric[9]

    
    def get_dataset_random(self, N_train, D):
        """
        Puntos interiores para la pérdida PDE (excluye bordes), generados de forma uniforme
        en cualquier dominio. Mantiene la misma interfaz que antes (N_train, D).
        Devuelve:
          - pts_pde: tensor (N_train, D)
        """
        # Generar puntos uniformes en [0,1]^D y escalarlos al dominio correspondiente
        pts = np.random.uniform(size=(N_train, D))
        if D == 2:
            mins = np.array([self.x_min, self.y_min])
            maxs = np.array([self.x_max, self.y_max])
        else:
            mins = np.full(D, self.x_min)
            maxs = np.full(D, self.x_max)
        pts = pts * (maxs - mins) + mins
        return torch.tensor(pts, dtype=torch.float32)
    
    
    def cartesian_combine(self, X: torch.Tensor, MU: torch.Tensor) -> torch.Tensor:
        """
        Devuelve todas las combinaciones cartesianas entre puntos espaciales X y parámetros MU.

        Args:
            X: tensor (N1, 2) con puntos espaciales [x1, x2]
            MU: tensor (N2, 2) con parámetros [mu1, mu2]

        Returns:
            tensor (N1*N2, 4) con columnas [x1, x2, mu1, mu2]
        """
        device = X.device
        dtype = X.dtype
        MU = MU.to(device=device, dtype=dtype)

        N1 = X.shape[0]
        N2 = MU.shape[0]

        # Expande y repite
        X_exp = X.unsqueeze(1).repeat(1, N2, 1)    # (N1, N2, 2)
        MU_exp = MU.unsqueeze(0).repeat(N1, 1, 1)  # (N1, N2, 2)

        combined = torch.cat([X_exp, MU_exp], dim=2)  # (N1, N2, 4)
        num_features = X.shape[1] + MU.shape[1]
        return combined.view(-1, num_features)
    
    
    def boundary_random_Dd(self,N_xx2, D):
        """
        Genera puntos en la frontera del dominio [x_min, x_max]^D usando muestreo uniforme aleatorio.

        Args:
            N_xx2 (int): Número total de puntos en la frontera.
            D (int): Número de dimensiones espaciales.
            x_min (float): Límite inferior del dominio.
            x_max (float): Límite superior del dominio.

        Returns:
            torch.Tensor: Puntos frontera de tamaño (N_bc, D).
        """
        N_dim_esp = int(N_xx2 / (2 * D))
        all_points = []
        for i in range(D):
            # puntos aleatorios uniformes en todo el cubo [x_min, x_max]^D
            points_min = np.random.uniform(low=self.x_min, high=self.x_max, size=(N_dim_esp, D))
            points_max = np.random.uniform(low=self.x_min, high=self.x_max, size=(N_dim_esp, D))

            # fijamos la i-ésima coordenada en el límite correspondiente
            points_min[:, i] = self.x_min
            points_max[:, i] = self.x_max

            all_points.append(points_min)
            all_points.append(points_max)

        # Juntamos todo y convertimos a tensor
        CC_points_np = np.vstack(all_points)
        CC_points = torch.tensor(CC_points_np, dtype=torch.float32)

        return CC_points
    
    def get_parametric_dataset(self, N_train, P):
        pts = np.random.uniform(size=(N_train, P))
        mins = np.array([self.mu1_min, self.mu2_min, self.beta_min, self.sigma1_min, self.sigma2_min])
        maxs = np.array([self.mu1_max, self.mu2_max, self.beta_max, self.sigma1_max, self.sigma2_max])
        pts = pts * (maxs - mins) + mins
        return torch.tensor(pts, dtype=torch.float32)

    def get_train_points(self, N_pde, D, N_mu, P, device=None):
        """
        Docstring for get_train_dataset
        Esta función se encarga de generar el conjunto de datos para el entrenamiento, 
        combinando los puntos del dominio físico y los del espacio paramétrico.
         - Para los puntos del dominio físico, se generan de forma aleatoria dentro del dominio definido por x_min, x_max, y_min, y_max.
         - Para los puntos del espacio paramétrico, se generan de forma aleatoria dentro del rango definido por mu1_min, mu1_max, mu2_min, mu2_max.
         - Luego, se combinan ambos conjuntos de puntos utilizando la función cartesian_combine para obtener el conjunto final de entrenamiento.

         Entradas:
            - N_pde: Número de puntos a generar en el dominio físico.
            - D: Número de dimensiones espaciales.
            - N_mu: Número de puntos a generar en el espacio paramétrico.
            - P: Número de dimensiones paramétricas.

        Salida:
            - combined_pts: Tensor de tamaño (N_pde * N_mu, D + P) que contiene todas las combinaciones cartesianas de los puntos del dominio físico y del espacio paramétrico.
        """
        pts_pde = self.get_dataset_random(N_pde, D=D).to(device=device, dtype=torch.float32)  # Puntos en el dominio físico
        # Generar puntos en el espacio paramétrico
        pts_parametric = self.get_parametric_dataset(N_mu, P=P).to(device=device, dtype=torch.float32)        # Combinar ambos conjuntos de puntos
        combined_pts = self.cartesian_combine(pts_pde, pts_parametric).to(device=device, dtype=torch.float32)  # Puntos combinados para entrenamiento
        return pts_pde, pts_parametric, combined_pts
    


    def get_test_dataset(self, N_test_x, N_test_y):
        """
        Devuelve una malla para testeo.
        """
        x = torch.linspace(self.x_min, self.x_max, N_test_x)
        y = torch.linspace(self.y_min, self.y_max, N_test_y)
        X, Y = torch.meshgrid(x, y, indexing='ij')  # Usar 'ij' para mantener orden correcto

        x_test = torch.hstack((X.reshape(-1,1), Y.reshape(-1,1)))  # (N_test_x * N_test_y, 2)

        return x_test, X, Y
