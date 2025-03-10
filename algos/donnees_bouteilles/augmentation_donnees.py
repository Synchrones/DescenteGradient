from PIL import Image
import numpy as np
import rembg
import tqdm
import matplotlib.pyplot as plt
import random


def bg_remove(infilename):
    img = Image.open( infilename )
    img.load()
    img = rembg.remove(img)
    return img


def alea(image) :
    img_bout = image
    img_bout.load()
    img_bout = img_bout.resize((200, 200))
    size_factor = random.uniform(0.85, 2.5)
    new_size = (int(200 * size_factor), int(200 * size_factor))
    img = Image.new("RGBA", (new_size[0], new_size[1]), (255, 255, 255, 0))
    img.paste(img_bout)
    img = img.rotate(random.randint(0, 360))
    img = img.resize((200, 200))
    data = np.asarray(img, dtype="int32")
    return data


def bg_bleu(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j][3] == 0:
                data[i][j][2], data[i][j][3] = (255, 255)
                

def tache_1(data : np.array, longueur_max, hauteure):
    taille = random.randint(10, longueur_max)
    depart_x = random.randint(1, data.shape[1] - taille - 1)
    depart_y = random.randint(1, data.shape[0] - hauteure - 1)
    random_blanc = random.randint(1, 150)
    rand_bleu = (random_blanc, random_blanc, 255 - random.randint(1, 75), 255)
    for i in range(hauteure):
        taille_x = random.randint(5, taille)
        for j in range(taille_x):
            dx = random.randint(0, 2 * (i + 1))
            if data[depart_y - i][depart_x + j - dx][0] == 0 and data[depart_y - i][depart_x + j - dx][1] == 0 and data[depart_y - i][depart_x + j - dx][2]  == 255:
                data[depart_y - i][depart_x + j - dx] = rand_bleu


if __name__ == '__main__':
    datas = []
    image = bg_remove('sticker-recyclage-bouteille-plastique-eau.jpg')
    for _ in tqdm.tqdm(range(25)):
        datas.append(alea(image))
    bouteilles = []
    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i in tqdm.tqdm(range(25)):
        bg_bleu(datas[i])
        for _ in range(20):
            tache_1(datas[i], random.randint(20, 50), random.randint(4, 20))
        bouteilles.append(datas[i])
    for i, ax in enumerate(axes.flat):
        ax.imshow(bouteilles[i])
        ax.axis('off')
    plt.show()