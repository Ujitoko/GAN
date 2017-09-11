import matplotlib.pyplot as plt
import os

from PIL import Image
import numpy as np
from enum import Enum

# save metrics
def save_metrics(model_name, metrics, epoch=None):
    # make directory if there is not
    dir_path = "metrics_" + model_name
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # save metrics
    plt.figure(figsize=(10,8))
    plt.plot(metrics["d_loss"], label="discriminative loss", color="b")
    plt.legend()
    plt.savefig(os.path.join(dir_path, "dloss" + str(epoch) + ".png"))
    plt.close()

    plt.figure(figsize=(10,8))
    plt.plot(metrics["g_loss"], label="generative loss", color="r")
    plt.legend()
    plt.savefig(os.path.join(dir_path, "g_loss" + str(epoch) + ".png"))
    plt.close()

    plt.figure(figsize=(10,8))
    plt.plot(metrics["g_loss"], label="generative loss", color="r")
    plt.plot(metrics["d_loss"], label="discriminative loss", color="b")
    plt.legend()
    plt.savefig(os.path.join(dir_path, "both_loss" + str(epoch) + ".png"))
    plt.close()

# save images
def save_imgs(model_name, images, plot_dim=(10,10), size=(10,10), name=None):
    # make directory if there is not
    dir_path = "generated_figures_" + model_name
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    num_examples = plot_dim[0]*plot_dim[1]
    num_examples = 100
    fig = plt.figure(figsize=size)

    for i in range(num_examples):
        plt.subplot(plot_dim[0], plot_dim[1], i+1)
        img = images[i, :]
        if img.shape[2] == 1:
            img = img.reshape((28, 28))
            plt.imshow(img, cmap="gray")
        else:
            #img = img.reshape((28, 28))
            plt.imshow(img)
        plt.tight_layout()
        plt.axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join(dir_path, str(name) + ".png"))
    plt.close()


class Label(Enum):
    cobwebbed = 0
    bubbly = 1
    chequered = 2
    gauzy = 3
    zigzagged = 4
    stratified = 5
    paisley = 6
    spiralled = 7
    blotchy = 8
    swirly = 9
    perforated = 10
    bumpy = 11
    veined = 12
    grid = 13
    braided = 14
    stained = 15
    dotted = 16
    matted = 17
    interlaced = 18
    potholed = 19
    striped = 20
    wrinkled = 21
    studded = 22
    meshed = 23
    woven = 24
    lined = 25
    flecked = 26
    polka_dotted = 27
    cracked = 28
    freckled = 29
    smeared = 30
    crosshatched = 31
    marbled = 32
    pleated = 33
    banded = 34
    grooved = 35
    knitted = 36
    pitted = 37
    waffled = 38
    fibrous = 39
    honeycombed = 40
    scaly = 41
    frilly = 42
    porous = 43
    sprinkled = 44
    lacelike = 45
    crystalline = 46


class DTD:
    def __init__(self):
        print("init DTD")
        with open("./DTD_128/labels/labels_joint_anno.txt", "r") as file:
            data = file.read()
            self.texture_len = len(data.split("\n"))
            self.texture = np.array(data.split("\n"))
            print("datasize of texture:{0}".format(self.texture_len))

    def extract(self, num, size):
        rand_index = np.random.randint(0, self.texture_len-1, size=num)
        ex_texture = self.texture[rand_index]

        tex_img = []
        tex_img_np = np.empty((0, size, size, 3), np.float32)
        label_np = np.empty((0, 47), np.float32)

        for (i, tex) in enumerate(ex_texture):
            # add tex_img
            tex_list = tex.split(" ")

            img = Image.open(os.path.join("./DTD_128/images", tex_list[0]))
            img_np = np.array(img)
            img_np = img_np[np.newaxis, :]
            tex_img_np = np.append(tex_img_np, img_np, axis=0)

            cat = np.zeros([1, 47]) -1
            for label in tex_list[1:-1]:
                label_ = label.replace("-", "_")
                index = Label[label_].value
                cat[:,index] = 1
            cat_ = cat[np.newaxis, :]
            label_np = np.append(label_np, cat, axis=0)

        (tex_img_np/255)
        return tex_img_np, label_np
