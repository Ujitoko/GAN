
# coding: utf-8

# # GAN implementation by tensorflow

# In[1]:

import tensorflow as tf
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# # utility関数
# 

# In[2]:

def save_metrics(metrics, epoch=None):
    plt.figure(figsize=(10,8))
    plt.plot(metrics["dis_loss"], label="discriminative loss", color="b")
    plt.legend()
    plt.savefig(os.path.join("metrics", "dloss" + str(epoch) + ".png"))
    plt.close()

    plt.figure(figsize=(10,8))
    plt.plot(metrics["gen_loss"], label="generative loss", color="r")
    plt.legend()
    plt.savefig(os.path.join("metrics", "g_loss" + str(epoch) + ".png"))
    plt.close()

    plt.figure(figsize=(10,8))
    plt.plot(metrics["gen_loss"], label="generative loss", color="r")
    plt.plot(metrics["dis_loss"], label="discriminative loss", color="b")
    plt.legend()
    plt.savefig(os.path.join("metrics", "both_loss" + str(epoch) + ".png"))
    plt.close()


# In[3]:

# noise[[examples, 100]]から生成した画像をplot_dim(例えば4x4)で表示
def save_imgs(images, plot_dim=(5,12), size=(12,5), epoch=None):
    examples = plot_dim[0]*plot_dim[1]

    # 表示
    fig = plt.figure(figsize=size)
    for i in range(examples):
        plt.subplot(plot_dim[0], plot_dim[1], i+1)
        img = images[i, :]
        img = img.reshape((96, 96, 3))
        plt.tight_layout()
        plt.imshow(img)
        plt.axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(os.path.join("generated_figures", str(epoch) + ".png"))
    plt.close()


# # モデル

# In[4]:

class Generator:
    def __init__(self):
        self.reuse = False
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.X_dim = 96*96*3
        self.h_dim = 128
        self.z_dim = 100

        self.depths = [1024, 512, 256, 128, 3]
        self.s_size = 6

    def __call__(self, inputs, training=False):
        with tf.variable_scope('g', reuse=self.reuse):
            # reshape from inputs
            outputs = tf.reshape(inputs, [-1, self.z_dim])
            
            with tf.variable_scope('reshape'):
                outputs = tf.layers.dense(outputs, self.depths[0] * self.s_size * self.s_size)
                outputs = tf.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            # deconvolution (transpose of convolution) x 4
            with tf.variable_scope('deconv1'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv2'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv3'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tf.nn.relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('deconv4'):
                outputs = tf.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            # output images
            with tf.variable_scope('tanh'):
#                outputs = tf.tanh(outputs, name='outputs')
                outputs = tf.sigmoid(outputs)
               # outputs = tf.reshape(outputs, [-1, 96, 96, 3])

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs


# In[5]:

class Discriminator:
    def __init__(self, batch_size=128):
        self.reuse = False
        self.h_dim = 128
        self.z_dim = 100
        self.X_dim = 96*96*3
        self.initializer = tf.contrib.layers.xavier_initializer()

        self.depths = [3, 64, 128, 256, 512]
        self.s_size = 6
        self.batch_size = batch_size

    def __call__(self, inputs, training=False, name=''):
        def leaky_relu(x, leak=0.2, name='outputs'):
            return tf.maximum(x, x * leak, name=name)

        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4
            outputs = tf.reshape(inputs, [-1, 96, 96, 3])
            with tf.variable_scope('conv1'):
                outputs = tf.layers.conv2d(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('conv2'):
                outputs = tf.layers.conv2d(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('conv3'):
                outputs = tf.layers.conv2d(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('conv4'):
                outputs = tf.layers.conv2d(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
                outputs = leaky_relu(tf.layers.batch_normalization(outputs, training=training), name='outputs')
            with tf.variable_scope('classify'):
                #batch_size = outputs.get_shape()[0].value
                #reshape = tf.reshape(outputs, [self.batch_size, -1])
                outputs = tf.layers.dense(outputs, 1, name='outputs')
                #outputs = tf.nn.sigmoid(outputs)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        return outputs


# In[6]:

class DCGAN:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.g = Generator()
        self.d = Discriminator()
        self.z_dim = 100

    def d_loss_calc(self, g_outputs, t_outputs, gradient_penalty):
        lambda_ = 10

        d_loss_train = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=t_outputs, labels=tf.ones_like(t_outputs)))
        d_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_outputs, labels=tf.zeros_like(g_outputs)))
        loss = d_loss_train + d_loss_gen
        loss = loss + lambda_*gradient_penalty
#        loss = -tf.reduce_mean(tf.log(t_outputs) + tf.log(1. - g_outputs)) + lambda_*gradient_penalty
        return loss

    def g_loss_calc(self, g_outputs):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=g_outputs, labels=tf.ones_like(g_outputs)))
        return loss

    def loss(self, traindata, perturbed_minibatch):
        z = tf.random_uniform([self.batch_size, self.z_dim], minval=-1.0, maxval=1.0)
        generated = self.g(z, training=True)
        g_outputs = self.d(generated, training=True, name='g')
        t_outputs = self.d(traindata, training=True, name='t')
        
        alpha = tf.random_uniform(shape=[self.batch_size,1], minval=0., maxval=1.)
        

        diff = perturbed_minibatch - traindata
        interpolates = traindata + (alpha*diff)
        gradients = tf.gradients(self.d(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes-1.)**2)
        #disc_cost += lambd*gradient_penalty
        
        # d_losses

        # add each losses to collection
        tf.add_to_collection(
            'g_losses',
            self.g_loss_calc(g_outputs))

        tf.add_to_collection(
            'd_losses',
            self.d_loss_calc(g_outputs, t_outputs, gradient_penalty)
        )

        return {
            self.g: tf.add_n(tf.get_collection('g_losses'), name='total_g_loss'),
            self.d: tf.add_n(tf.get_collection('d_losses'), name='total_d_loss'),
        }

    def d_train(self, losses, learning_rate=1e-4, beta1=0.5, beta2=0.9):
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        d_opt_op = d_opt.minimize(losses[self.d], var_list=self.d.variables)
        return d_opt_op

    def g_train(self, losses, learning_rate=1e-4, beta1=0.5, beta2=0.9):
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2)
        g_opt_op = g_opt.minimize(losses[self.g], var_list=self.g.variables)
        return g_opt_op

    def sample_images(self, row=5, col=12, inputs=None, epoch=None):
        images = self.g(inputs, training=True)
        #save_imgs(images, epoch=epoch)
        return images


# In[7]:

#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# In[8]:

import numpy as np

X_train = np.load("irasutoya_face_1813x96x96x3_jpg.npy")
X_train = X_train/255
X_dim = 96*96*3
z_dim = 100
batch_size = 128
epochs = 500000
display_epoch = 100
param_save_epoch = 10000
loss = {"dis_loss":[], "gen_loss":[]}


# In[9]:

p_noise = tf.placeholder(tf.float32, [None, z_dim])
noise_check = np.random.uniform(-1, 1, size=[60, z_dim]).astype(np.float32)

X = tf.placeholder(tf.float32, [None, X_dim])
X_p = tf.placeholder(tf.float32, [None, X_dim])

dcgan = DCGAN()
losses = dcgan.loss(traindata=X, perturbed_minibatch=X_p)
d_train_op = dcgan.d_train(losses)
g_train_op = dcgan.g_train(losses)

saver = tf.train.Saver()
#%debug
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        if epoch % display_epoch == 0:
            imgs_gen = dcgan.sample_images(inputs=noise_check).eval()
            print("saving images")
            save_imgs(imgs_gen, epoch=epoch)
        for _ in range(10):
            # ノイズから生成データを生成
            #noise = np.random.uniform(-1, 1, size=[batch_size, z_dim])
            # 訓練データを抜粋
            rand_index = np.random.randint(0, X_train.shape[0], size=batch_size)
            X_mb = X_train[rand_index, :].astype(np.float32)
            X_mb = X_mb.reshape([-1, X_dim])
            #X_mb, _ = mnist.train.next_batch(batch_size)
            
            # x + delta vector
            X_mb_p = X_mb + 0.5*X_mb.std()*np.random.random(X_mb.shape)

            _, d_loss_value = sess.run([d_train_op, losses[dcgan.d]], feed_dict={X: X_mb, X_p:X_mb_p})

        _, g_loss_value, = sess.run([g_train_op, losses[dcgan.g]], feed_dict={X: X_mb, X_p:X_mb_p})

        # 結果をappend
        loss["dis_loss"].append(d_loss_value)
        loss["gen_loss"].append(g_loss_value)
        print("epoch:" + str(epoch))
        # グラフの描画（余裕があったら）
        if epoch % display_epoch == 0:
            save_metrics(loss, epoch)

        if epoch % param_save_epoch == 0:
            saver.save(sess, "./model/dcgan_model" + str(epoch) + ".ckpt")


# In[ ]:

import numpy as np

a = np.arange(10)
print(a)
print(a.shape)

b = a - 1
print(b)
print(b.shape)
print(np.abs(b))
#np.abs
#tf.pow((tf.abs(delta)-1), 2))


# In[ ]:

a = [[[0],[1],[2]],
     [[0],[1],[2]],
     [[0],[1],[2]],
     [[0],[1],[2]]]

b = tf.reduce_sum(a, axis=0)

with tf.Session() as sess:
    c = sess.run(b)
    
print(np.array(a).shape)
print(c)


# In[ ]:



