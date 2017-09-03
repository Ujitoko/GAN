import matplotlib.pyplot as plt
import os

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
