#!/bin/bash
# Script optimizado para PyTorch en FT3 (CESGA)

# =================================================================
# 1. SOLICITUD DE RECURSOS (SLURM Directives)
# =================================================================
# Usamos la partición genérica 'gpu' y forzamos los límites requeridos:
# Requisito FT3: 32 CPUs por 1 GPU A100. [cite: 737, 744, 750, 807, 1551]
# Tiempo máximo para la partición short (que incluye GPU) es 06:00:00 [cite: 782, 795, 804, 825, 826]
# GRES: Solicitamos 1 GPU explícitamente [cite: 738, 744, 1552, 1553, 5455]

#SBATCH --job-name=p-pinns_Training
#SBATCH --partition=gpu             
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # OBLIGATORIO: 32 cores por GPU.
#SBATCH --time=00:40:00             
#SBATCH --mem=30G                   
#SBATCH --gres=gpu:1                # Solicita 1 GPU
#SBATCH -o logs/download_data.log -e logs/download_data_error.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=miguel.delasheras@usc.es


# =================================================================
# 2. CONFIGURACIÓN DEL ENTORNO
# (Usando la versión de miniconda validada y la variable $STORE)
# =================================================================
module purge
module load cesga/system miniconda3/22.11.1-1

# Activa tu entorno PyTorch en el Store. Usa la variable $STORE para la ruta.
conda activate $STORE/conda_envs/deep

# =================================================================
# 3. EJECUCIÓN DEL CÓDIGO
# (Añadimos el 'cd' que faltaba para asegurar la ruta de ejecución)
# =================================================================

echo "Iniciando descarga"
# Ejecuta el script principal de Python
python main.py

# Desactiva el entorno
conda deactivate