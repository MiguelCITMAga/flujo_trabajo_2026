import os
from PIL import Image
import imageio
import glob

def generate_gif(test_folder, count, remove_imgs=True):
    fps = max(1, count // 5)

    # ----- GIF de la solución 3D -----
    imgs = []
    img_files = sorted(glob.glob(os.path.join(test_folder, "approx_*.png")))

    for img_file in img_files:
        img = imageio.imread(img_file)
        img = Image.fromarray(img)
        imgs.append(img)
        if remove_imgs:
            os.remove(img_file)

    if len(imgs) > 0:
        imageio.mimwrite(os.path.join(test_folder, "0_video_u.mp4"), imgs, fps=fps)
        imageio.mimsave(os.path.join(test_folder, "0_video_u.gif"), imgs, format='GIF')
    else:
        print("⚠️ No se encontraron imágenes 3D para generar el GIF.")

    # ----- GIF del mapa de calor (contornos) -----
    imgs = []
    contour_files = sorted(glob.glob(os.path.join(test_folder, "contour_approx_*.png")))

    for img_file in contour_files:
        img = imageio.imread(img_file)
        img = Image.fromarray(img)
        imgs.append(img)
        if remove_imgs:
            os.remove(img_file)

    if len(imgs) > 0:
        imageio.mimwrite(os.path.join(test_folder, "0_video_u_contour.mp4"), imgs, fps=fps)
        imageio.mimsave(os.path.join(test_folder, "0_video_u_contour.gif"), imgs, format='GIF')
    else:
        print("⚠️ No se encontraron mapas de calor para generar el GIF de contornos.")
