import os
import time
from matplotlib import gridspec, pyplot as plt
import numpy as np
import pytz
import torch

# Funciones auxiliares que necesitamos
def create_tests_folder(parent_folder="", prefix="", postfix=""):
    """
    Crea una carpeta para guardar resultados
    """
    time_stamp = int(time.time())
    time_zone = pytz.timezone("Europe/Berlin")
    test_time = pytz.datetime.datetime.fromtimestamp(time_stamp, time_zone)
    test_time = test_time.strftime("%Y%m%d-%H%M%S")
    tests_folder = os.getcwd() + f"/{parent_folder}/test{prefix}_{test_time}{postfix}"
    os.makedirs(tests_folder)
    print(f"\nWorking in folder {tests_folder}\n")
    return tests_folder



