import torch
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, layers, domain):
        super().__init__()
        self.layers = layers
        self.domain = domain
        self.activation = nn.Tanh()
        
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])

        for i in range(len(self.layers) - 1):
            #nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.kaiming_normal_(self.linears[i].weight.data, nonlinearity='relu')
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        eta = (1 - x[:,0:1]**2) * (1 - x[:,1:2]**2) # hard constraints para imponer contorno
        return a * eta

    def loss_bc(self, x_in):
        # Pérdida para condiciones de contorno (Dirichlet homogéneo)
        x_in = x_in.clone().requires_grad_(True)
        u_data = self.forward(x_in)
        bc_loss = torch.mean(u_data**2)  # Queremos que u sea 0 en el borde
        return bc_loss


    def loss_pde(self, x_in, domain_volume=1.0):
        x_in = x_in.clone().requires_grad_(True)
        # 1. Separamos las variables
        x = x_in[:, 0:1]   # (N, 1)
        y = x_in[:, 1:2]
        k1       = x_in[:, 2:3] # (N, 1)
        k2       = x_in[:, 3:4] # (N, 1)
        beta = x_in[:, 4:5]
        sigma1 = x_in[:, 5:6]
        sigma2 = x_in[:, 6:7]
        u_data = self.forward(x_in)       # (N, 1)
        # 2. Calculamos gradiente solo respecto a x_spatial
        # (PyTorch calculará gradiente respecto a todo x_in, tomamos la columna 0)
        grads = torch.autograd.grad(
            u_data, x_in,
            grad_outputs=torch.ones_like(u_data),
            create_graph=True,
            retain_graph=True
        )[0]
        
        grads_spatial = grads[:, 0:2] # Derivada parcial du/dx

        # 3. Definimos k(x) dinámicamente según los parámetros de cada fila
        # k = k1 si x < x_hat, sino k2
        k_val = torch.where(x < beta, k1, k2)
        f = torch.exp(-2*((x_in[:,0:1] - sigma1)**2 + (x_in[:,1:2] - sigma2)**2))
        # 4. Funcional de energía: Integral [ 0.5 * k * (u')^2 - f * u ]
        # Nota: f_data asumimos que depende solo de x (ej: sin(x))
        grad_norm_u_squared = torch.sum(grads_spatial**2, dim=1).unsqueeze(1) # (N, 1)
        a_u_u = domain_volume * (torch.mean(k_val * grad_norm_u_squared))
        l_u = domain_volume * torch.mean(f * u_data)
        energy_density = 0.5 * a_u_u - l_u

        return energy_density
    
    def total_loss(self, x_pde, x_bc, x_hat=1/2, domain_volume=1.0, lambda_bc=1.0):
        loss_pde = self.loss_pde(x_pde, x_hat=x_hat, domain_volume=domain_volume)
        loss_bc = self.loss_bc(x_bc)
        return loss_pde + lambda_bc * loss_bc
    
class FNN_comb(nn.Module):
    def __init__(self, layers_x, layers_mu):
        super().__init__()
        self.layers_x = layers_x
        self.layers_mu = layers_mu

        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction="mean")

        self.linears_x = nn.ModuleList([
            nn.Linear(layers_x[i], layers_x[i+1])
            for i in range(len(layers_x) - 1)
        ])

        self.linears_mu = nn.ModuleList([
            nn.Linear(layers_mu[i], layers_mu[i+1])
            for i in range(len(layers_mu) - 1)
        ])

        for layer in self.linears_x:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        for layer in self.linears_mu:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Añadimos un sesgo aprendible en la capa de salida de la red combinada
        self.final_bias = nn.Parameter(torch.zeros(1))

    # ---------- Redes separables ----------
    def forward_x(self, x):
        a = x
        for i in range(len(self.layers_x) - 2):
            a = self.activation(self.linears_x[i](a))
        return self.linears_x[-1](a)  # (N, r)

    def forward_mu(self, mu):
        a = mu
        for i in range(len(self.layers_mu) - 2):
            a = self.activation(self.linears_mu[i](a))
        return self.linears_mu[-1](a)  # (N, r)

    # ---------- Forward conjunto ----------
    def forward(self, x, mu):
        # x:  (N,2)
        # mu: (N,d_mu)
        x_feat = self.forward_x(x)     # (N, r)
        mu_feat = self.forward_mu(mu)  # (N, r)

        u = torch.sum(x_feat * mu_feat, dim=1, keepdim=True)  #+ self.final_bias # (N,1)

        eta = (1 - x[:,0:1]**2) * (1 - x[:,1:2]**2)
        return u * eta

    # ---------- Loss ----------
    def loss(self, x_phys, mu_param, domain_volume=1.0):
        Nx = x_phys.shape[0]
        Nmu = mu_param.shape[0]
        x_phys = x_phys.unsqueeze(1).expand(Nx, Nmu, 2).reshape(-1, 2)
        mu_param = mu_param.unsqueeze(0).expand(Nx, Nmu, 5).reshape(-1, 5)

        x_phys = x_phys.clone().requires_grad_(True)
        # 1. Separamos las variables
        x = x_phys[:, 0:1]   # (N, 1)
        y = x_phys[:, 1:2]
        k1       = mu_param[:, 0:1] # (N, 1)
        k2       = mu_param[:, 1:2] # (N, 1)
        beta = mu_param[:, 2:3]
        sigma1 = mu_param[:, 3:4]
        sigma2 = mu_param[:, 4:5]
        u_data = self.forward(x_phys, mu_param)       # (N, 1)
        # 2. Calculamos gradiente solo respecto a x_spatial
        # (PyTorch calculará gradiente respecto a todo x_in, tomamos la columna 0)
        grads = torch.autograd.grad(
            u_data, x_phys,
            grad_outputs=torch.ones_like(u_data),
            create_graph=True,
            retain_graph=True
        )[0]
        
        grads_spatial = grads[:, 0:2] # Derivada parcial du/dx

        # 3. Definimos k(x) dinámicamente según los parámetros de cada fila
        # k = k1 si x < x_hat, sino k2
        k_val = torch.where(x < beta, k1, k2)
        f = torch.exp(-2*((x_phys[:,0:1] - sigma1)**2 + (x_phys[:,1:2] - sigma2)**2))
        # 4. Funcional de energía: Integral [ 0.5 * k * (u')^2 - f * u ]
        # Nota: f_data asumimos que depende solo de x (ej: sin(x))
        grad_norm_u_squared = torch.sum(grads_spatial**2, dim=1).unsqueeze(1) # (N, 1)
        a_u_u = domain_volume * (torch.mean(k_val * grad_norm_u_squared))
        l_u = domain_volume * torch.mean(f * u_data)
        energy_density = 0.5 * a_u_u - l_u

        return energy_density

class PGD_Separable_PINN(nn.Module):
    def __init__(self, layers_x, layers_mu):
        super().__init__()
        self.layers_x = layers_x
        self.layers_mu = layers_mu
        self.activation = nn.Tanh()

        # Listas dinámicas para guardar las capas de cada modo (término de la suma)
        self.modos_x = nn.ModuleList()
        self.modos_mu = nn.ModuleList()

    def add_mode(self):
        """Crea un nuevo modo separable y lo añade a la arquitectura global."""
        # Creamos las capas para el nuevo modo
        linears_x = nn.ModuleList([
            nn.Linear(self.layers_x[i], self.layers_x[i+1])
            for i in range(len(self.layers_x) - 1)
        ])

        linears_mu = nn.ModuleList([
            nn.Linear(self.layers_mu[i], self.layers_mu[i+1])
            for i in range(len(self.layers_mu) - 1)
        ])

        # Inicializamos los pesos del nuevo modo
        for layer in linears_x:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        for layer in linears_mu:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

        # Los añadimos a la lista global de la red
        self.modos_x.append(linears_x)
        self.modos_mu.append(linears_mu)

    def forward(self, x, mu):
        u_total = 0
        
        # Iteramos sobre todos los modos que se hayan añadido hasta ahora
        for mode_x, mode_mu in zip(self.modos_x, self.modos_mu):
            # Forward de la rama espacial para este modo
            a_x = x
            for layer in mode_x[:-1]:
                a_x = self.activation(layer(a_x))
            out_x = mode_x[-1](a_x)

            # Forward de la rama paramétrica para este modo
            a_mu = mu
            for layer in mode_mu[:-1]:
                a_mu = self.activation(layer(a_mu))
            out_mu = mode_mu[-1](a_mu)

            # Producto y suma a la predicción total
            u_total = u_total + torch.sum(out_x * out_mu, dim=1, keepdim=True)

        eta = (1 - x[:,0:1]**2) * (1 - x[:,1:2]**2)
        return u_total * eta

    def loss(self, x_phys, mu_param, domain_volume=1.0):
        # ---------- TU LOSS EXACTAMENTE COMO LA TENÍAS ----------
        Nx = x_phys.shape[0]
        Nk = mu_param.shape[0]
        x_phys = x_phys.unsqueeze(1).expand(Nx, Nk, 2).reshape(-1, 2)
        mu_param = mu_param.unsqueeze(0).expand(Nx, Nk, 5).reshape(-1, 5)

        x_phys = x_phys.clone().requires_grad_(True)
        
        x = x_phys[:, 0:1]   
        y = x_phys[:, 1:2]
        k1 = mu_param[:, 0:1] 
        k2 = mu_param[:, 1:2] 
        beta = mu_param[:, 2:3]
        sigma1 = mu_param[:, 3:4]
        sigma2 = mu_param[:, 4:5]
        
        # Llama internamente al forward, que ahora suma todos los modos
        u_data = self.forward(x_phys, mu_param)       
        
        grads = torch.autograd.grad(
            u_data, x_phys,
            grad_outputs=torch.ones_like(u_data),
            create_graph=True,
            retain_graph=True
        )[0]
        
        grads_spatial = grads[:, 0:2] 

        k_val = torch.where(x < beta, k1, k2)
        f = torch.exp(-2*((x_phys[:,0:1] - sigma1)**2 + (x_phys[:,1:2] - sigma2)**2))
        
        grad_norm_u_squared = torch.sum(grads_spatial**2, dim=1).unsqueeze(1) 
        a_u_u = domain_volume * (torch.mean(k_val * grad_norm_u_squared))
        l_u = domain_volume * torch.mean(f * u_data)
        energy_density = 0.5 * a_u_u - l_u

        return energy_density


def relu3(x):
    return torch.max(torch.zeros_like(x), x) ** 3

class ResNet(nn.Module):
    def __init__(self, layers, domain):
        super().__init__()
        self.layers = layers
        self.domain = domain
        self.activation = nn.Tanh()  # Puedes probar con otras activaciones como ReLU, Tanh, etc.
        
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(len(self.layers) - 1)])
        self.iter = 0

        # Capa de entrada
        self.input_layer = nn.Linear(layers[0], layers[1])

        # Crear pares de lineales para bloques residuales
        self.linear1_list = nn.ModuleList()
        self.linear2_list = nn.ModuleList()
        for _ in range(len(layers) - 3):
            self.linear1_list.append(nn.Linear(layers[1], layers[1]))
            self.linear2_list.append(nn.Linear(layers[1], layers[1]))

        # Capa de salida
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        #self.output_activation = nn.Softplus()

        # Inicialización Kaiming
        #nn.init.kaiming_normal_(self.input_layer.weight.data, nonlinearity='relu')
        nn.init.xavier_normal_(self.input_layer.weight.data, gain=1.0)
        nn.init.zeros_(self.input_layer.bias.data)
        
        for l1, l2 in zip(self.linear1_list, self.linear2_list):
            #nn.init.kaiming_normal_(l1.weight.data, nonlinearity='relu')
            nn.init.xavier_normal_(l1.weight.data, gain=1.0)
            nn.init.zeros_(l1.bias.data)
            #nn.init.kaiming_normal_(l2.weight.data, nonlinearity='relu')
            nn.init.xavier_normal_(l2.weight.data, gain=1.0)
            nn.init.zeros_(l2.bias.data)

        #nn.init.kaiming_normal_(self.output_layer.weight.data, nonlinearity='relu')
        nn.init.xavier_normal_(self.output_layer.weight.data, gain=1.0) 
        nn.init.zeros_(self.output_layer.bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        
        a = x.float()
        
        a = self.activation(self.input_layer(a))

        for l1, l2 in zip(self.linear1_list, self.linear2_list):
            residual = a
            out = self.activation(l1(a))
            out = self.activation(l2(out))
            a = out + residual

        a = self.output_layer(a)
        eta = (1 - x[:,0:1]**2) * (1 - x[:,1:2]**2) # hard constraints para imponer contorno

        #a = self.output_activation(a)
        return a * eta
    
    def loss_pde(self, x_in, domain_volume=1.0):
        x_in = x_in.clone().requires_grad_(True)
        # 1. Separamos las variables
        x = x_in[:, 0:1]   # (N, 1)
        y = x_in[:, 1:2]
        k1       = x_in[:, 2:3] # (N, 1)
        k2       = x_in[:, 3:4] # (N, 1)
        beta = x_in[:, 4:5]
        sigma1 = x_in[:, 5:6]
        sigma2 = x_in[:, 6:7]
        u_data = self.forward(x_in)       # (N, 1)
        # 2. Calculamos gradiente solo respecto a x_spatial
        # (PyTorch calculará gradiente respecto a todo x_in, tomamos la columna 0)
        grads = torch.autograd.grad(
            u_data, x_in,
            grad_outputs=torch.ones_like(u_data),
            create_graph=True,
            retain_graph=True
        )[0]
        
        grads_spatial = grads[:, 0:2] # Derivada parcial du/dx

        # 3. Definimos k(x) dinámicamente según los parámetros de cada fila
        # k = k1 si x < x_hat, sino k2
        k_val = torch.where(x < beta, k1, k2)
        f = torch.exp(-2*((x_in[:,0:1] - sigma1)**2 + (x_in[:,1:2] - sigma2)**2))
        # 4. Funcional de energía: Integral [ 0.5 * k * (u')^2 - f * u ]
        # Nota: f_data asumimos que depende solo de x (ej: sin(x))
        grad_norm_u_squared = torch.sum(grads_spatial**2, dim=1).unsqueeze(1) # (N, 1)
        a_u_u = domain_volume * (torch.mean(k_val * grad_norm_u_squared))
        l_u = domain_volume * torch.mean(f * u_data)
        energy_density = 0.5 * a_u_u - l_u

        return energy_density