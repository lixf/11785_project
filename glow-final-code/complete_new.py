#!/usr/bin/env python3

import os
import sys
import time

import horovod.tensorflow as hvd
import numpy as np
import tensorflow as tf
import graphics
from utils import ResultLogger
from utils_complete import *

from memory_saving_gradients import gradients as grad

learn = tf.contrib.learn

from ops import *


class GlowModel(object):

    def __init__(self, sess, model, image_size=64,
                 batch_size=64, sample_size=64,
                 checkpoint_dir=None, lam=0.1):

        self.image_size = image_size
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.lam = lam
        self.model = model
        self.sess = sess

        self.image_shape = [self.image_size, self.image_size, 3]

        self.gf_dim = 64
        log_size = int(math.log(image_size) / math.log(2))
        self.g_bns = [
            batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]

        self.build_model()

    def build_model(self):

        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        # self.zhats = tf.placeholder(
        #     tf.float32, [None, 4, 4, 96], name='zhats')

        self.zhats = tf.placeholder(
            tf.float32, [None, 8, 8, 48], name='zhats')

        #self.zhats = tf.placeholder(tf.float32, [None, 100], name='zhats')

        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.eps_std = tf.placeholder(tf.float32, [None], name='eps_std_comp')

        self.G_imgs = self.model.f_convert_sample_into_image(
            self.zhats, self.eps_std)

        #self.G_imgs = self.generator(self.zhats)

        self.labels = tf.placeholder(
            tf.int32, [1], name='labels')

        print(self.mask.shape, self.G_imgs.shape, self.images.shape)

        self.contextual_loss = tf.cast(tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.mask, self.G_imgs) - tf.multiply(self.mask, self.images))), 1), tf.float32)

        # self.perceptual_loss = self.model.loss_complete(
        #     self.G_imgs, np.array([0]), False, True)
        self.perceptual_loss = tf.convert_to_tensor(self.model.loss_complete(
            self.G_imgs, np.array([0]), False, True), dtype=tf.float32)

        self.complete_loss = self.contextual_loss + self.lam * self.perceptual_loss
        self.grad_complete_loss = tf.gradients(
            self.complete_loss, self.zhats)

    def complete(self, config, visualise, model):
        def make_dir(name):
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)

        make_dir('hats_imgs')
        make_dir('completed')
        make_dir('logs_complete')

        logdir = os.path.abspath(config.logdir) + "/"
        if not os.path.exists(logdir):
            os.mkdir(logdir)

        # image_shape = [config.imgSize, config.imgSize, 3]
        # image_size = config.imgSize
        # lam = 0.1

        # try:
        #     tf.global_variables_initializer().run(session=self.sess)
        # except:
        #     tf.initialize_all_variables().run()
        # print("reahching here")

        nImgs = len(config.imgs)
        batch_size = min(64, nImgs)
        batch_idxs = int(np.ceil(nImgs / batch_size))

        #mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        #zhats = tf.placeholder(tf.float32, [None, 4, 4, 48], name='z_hats')
        # images = tf.placeholder(
        #    tf.float32, [None] + self.image_shape, name='real_images')
        #is_training = tf.placeholder(tf.bool, name='is_training')

        if config.maskType == 'center':
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size * config.centerScale)
            u = int(self.image_size * (1.0 - config.centerScale))
            mask[l:u, l:u, :] = 0.0

        for idx in range(0, batch_idxs):
            l = idx * batch_size
            u = min((idx + 1) * batch_size, nImgs)
            batchSz = u - l
            batch_files = config.imgs[l:u]

            #print("Batching here", self.image_size)
            batch = [get_image(batch_file, self.image_size, is_crop=True)
                     for batch_file in batch_files]
            #print("Batching here", batch[0].shape)
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < batch_size:
                padSz = ((0, int(batch_size - batchSz)),
                         (0, 0), (0, 0), (0, 0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)
            #batch_images = self.postprocess(batch_images)

            zhats = visualise(batch_size)
            #zhats = np.random.uniform(-1, 1, size=(3, 100))
            #zhats = np.random.uniform(-1, 1, size=(3, 8, 8, 48))

            m = 0
            v = 0

            nRows = np.ceil(batchSz / 8)
            nCols = min(8, batchSz)
            save_images(batch_images[:batchSz, :, :, :], [nRows, nCols],
                        os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, mask)
            save_images(masked_images[:batchSz, :, :, :], [nRows, nCols],
                        os.path.join(config.outDir, 'masked.png'))

            #graphics.save_raster(batch_images,"before.png")

            for i in range(config.nIter):
                print("Iteration ", i)

                fd = {
                    self.zhats: zhats,
                    self.mask: mask,
                    self.images: batch_images,
                    self.eps_std: [1.0] * batch_size
                    # self.G_imgs: G_imgs
                }

    

                run = [self.complete_loss, self.grad_complete_loss, self.G_imgs]
                loss, g, G_imgs = self.sess.run(run, feed_dict=fd)

                #print(G_imgs)
                #save_images(G_imgs,[nRows, nCols], "image.png")
                #print("Loss is", type(G_imgs))

                

                # if i % 2 == 0:
                #     model.save(logdir + "model_best_loss.ckpt")

                if i % config.outInterval == 0:
                    print(i, np.mean(loss[0:batchSz]))
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:04d}.png'.format(i))
                    nRows = np.ceil(batchSz / 8)
                    nCols = min(8, batchSz)
                    save_images(G_imgs[:batchSz, :, :, :],
                                [nRows, nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0 - mask)
                    completed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:04d}.png'.format(i))
                    save_images(completed[:batchSz, :, :, :],
                                [nRows, nCols], imgName)

                if config.approach == 'adam':
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = config.beta1c * m_prev + (1 - config.beta1c) * g[0]
                    v = config.beta2c * v_prev + \
                        (1 - config.beta2c) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - config.beta1c ** (i + 1))
                    v_hat = v / (1 - config.beta2c ** (i + 1))
                    zhats += - np.true_divide(config.lr *
                                              m_hat, (np.sqrt(v_hat) + config.eps))
                # zhats = np.clip(zhats, -1, 1)

    def postprocess(self, x):
        n_bins = 256
        return np.clip(((x + 0.5) * 256), 0, 255)


def init_visualizations(hps, model, logdir):

    def sample_batch(y, eps):
        n_batch = hps.local_batch_train
        sample_data_list = []
        sample_points_list = []

        for i in range(int(np.ceil(len(eps) / n_batch))):
            sample_points = model.sample_complete(
                y[i * n_batch:i * n_batch + n_batch], eps[i * n_batch:i * n_batch + n_batch])
            # sample_data_list.append(sample_data)
            sample_points_list.append(sample_points)

        return np.concatenate(sample_points_list)

    def draw_samples(batch_size):
        if hvd.rank() != 0:
            return

        # rows = 10 if hps.image_size <= 64 else 4
        # cols = rows
        n_batch = batch_size
        y = np.asarray([_y % hps.n_y for _y in (
            range(n_batch))], dtype='int32')

        # y = np.asarray([0])
        # n_batch = 1

        # temperatures = [0., .25, .5, .626, .75, .875, 1.] #previously
        temperatures = [0., .25, .5, .6, .7, .8, .9, 1.]

        # x_samples = []
        # x_samples.append(sample_batch(y, [.0] * n_batch))
        # x_samples.append(sample_batch(y, [.25] * n_batch))
        # x_samples.append(sample_batch(y, [.5] * n_batch))
        # x_samples.append(sample_batch(y, [.6] * n_batch))
        # x_samples.append(sample_batch(y, [.7] * n_batch))
        # x_samples.append(sample_batch(y, [.8] * n_batch))
        # x_samples.append(sample_batch(y, [.9] * n_batch))
        # x_samples.append(sample_batch(y, [1.] * n_batch))
        # # previously: 0, .25, .5, .625, .75, .875, 1.

        # for i in range(len(x_samples)):
        #     x_sample = np.reshape(
        #         x_samples[i], (n_batch, hps.image_size, hps.image_size, 3))
        #     graphics.save_raster(x_sample, logdir +
        #                          'epoch_{}_sample_{}.png'.format(epoch, i))

        sample_points = sample_batch(y, [1.0] * n_batch)
        return sample_points

    return draw_samples


def main(hps):

    print("Init horovod")
    # Initialize Horovod.
    hvd.init()

    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hvd.rank() + hvd.size() * hps.seed)
    np.random.seed(hvd.rank() + hvd.size() * hps.seed)

    # Get data and set train_its and valid_its
    train_iterator, test_iterator, data_init = get_data(hps, sess)
    hps.train_its, hps.test_its, hps.full_test_its = get_its(hps)

    # for i in range(64):
    #     X, Y = train_iterator.get_next()
    #     #X, Y = train_iterator.get_next()
    #     X = X.eval(session=sess)
    #     X = X[0]
    #     X = np.expand_dims(X, axis=0)
    #     print(X.shape)

    #     graphics.save_raster(X, "./gen/abcd" + str(i) + ".png")

    # print("load horovod")

    # Create log dir
    logdir = os.path.abspath(hps.logdir) + "/"
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    # Create model
    import model
    model = model.model(sess, hps, train_iterator, test_iterator, data_init)

    # Initialize visualization functions
    visualise = init_visualizations(hps, model, logdir)

    xx = visualise(1)
    print(xx.shape)

    glow = GlowModel(sess, model, image_size=hps.imgSize,
                     batch_size=min(64, len(hps.imgs)),
                     checkpoint_dir=hps.checkpointDir, lam=hps.lam)

    glow.complete(hps, visualise, model)


def get_data(hps, sess):
    if hps.image_size == -1:
        hps.image_size = {'mnist': 32, 'cifar10': 32, 'imagenet-oord': 64,
                          'imagenet': 256, 'celeba': 256, 'lsun_realnvp': 64, 'lsun': 256}[hps.problem]
    if hps.n_test == -1:
        hps.n_test = {'mnist': 10000, 'cifar10': 10000, 'imagenet-oord': 50000, 'imagenet': 50000,
                      'celeba': 3000, 'lsun_realnvp': 300 * hvd.size(), 'lsun': 300 * hvd.size()}[hps.problem]
    hps.n_y = {'mnist': 10, 'cifar10': 10, 'imagenet-oord': 1000,
               'imagenet': 1000, 'celeba': 1, 'lsun_realnvp': 1, 'lsun': 1}[hps.problem]
    if hps.data_dir == "":
        hps.data_dir = {'mnist': None, 'cifar10': None, 'imagenet-oord': '/mnt/host/imagenet-oord-tfr', 'imagenet': '/mnt/host/imagenet-tfr',
                        'celeba': '/mnt/host/celeba-reshard-tfr', 'lsun_realnvp': '/mnt/host/lsun_realnvp', 'lsun': '/mnt/host/lsun'}[hps.problem]

    if hps.problem == 'lsun_realnvp':
        hps.rnd_crop = True
    else:
        hps.rnd_crop = False

    if hps.category:
        hps.data_dir += ('/%s' % hps.category)

    # Use anchor_size to rescale batch size based on image_size
    s = hps.anchor_size
    hps.local_batch_train = hps.n_batch_train * \
        s * s // (hps.image_size * hps.image_size)
    hps.local_batch_test = {64: 50, 32: 25, 16: 10, 8: 5, 4: 2, 2: 2, 1: 1}[
        hps.local_batch_train]  # round down to closest divisor of 50
    hps.local_batch_init = hps.n_batch_init * \
        s * s // (hps.image_size * hps.image_size)

    print("Rank {} Batch sizes Train {} Test {} Init {}".format(
        hvd.rank(), hps.local_batch_train, hps.local_batch_test, hps.local_batch_init))

    if hps.problem in ['imagenet-oord', 'imagenet', 'celeba', 'lsun_realnvp', 'lsun']:
        hps.direct_iterator = True
        import data_loaders.get_data as v
        train_iterator, test_iterator, data_init = \
            v.get_data(sess, hps.data_dir, hvd.size(), hvd.rank(), hps.pmap, hps.fmap, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size, hps.rnd_crop)

    elif hps.problem in ['mnist', 'cifar10']:
        hps.direct_iterator = False
        import data_loaders.get_mnist_cifar as v
        train_iterator, test_iterator, data_init = \
            v.get_data(hps.problem, hvd.size(), hvd.rank(), hps.dal, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size)

    else:
        raise Exception()

    return train_iterator, test_iterator, data_init


def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller,
    # we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())
    assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess


if __name__ == "__main__":

    # This enables a ctr-C without triggering errors
    # import signal
    # signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--restore_path", type=str, default='',
                        help="Location of checkpoint to restore")
    parser.add_argument("--inference", action="store_true",
                        help="Use in inference mode")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='celeba',
                        help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--category", type=str,
                        default='', help="LSUN category")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=-
                        1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=50, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=0,
                        help="Coupling type: 0=additive, 1=affine")

    # Complete argumentss

    parser.add_argument('--approach', type=str,
                        choices=['adam', 'hmc'],
                        default='adam')
    # parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1c', type=float, default=0.9)
    parser.add_argument('--beta2c', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--hmcBeta', type=float, default=0.2)
    parser.add_argument('--hmcEps', type=float, default=0.001)
    parser.add_argument('--hmcL', type=int, default=100)
    parser.add_argument('--hmcAnneal', type=float, default=1)
    parser.add_argument('--nIter', type=int, default=1000)
    parser.add_argument('--imgSize', type=int, default=64)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--checkpointDir', type=str, default='checkpoint')
    parser.add_argument('--outDir', type=str, default='completions')
    parser.add_argument('--outInterval', type=int, default=1)
    parser.add_argument('--maskType', type=str,
                        choices=['random', 'center', 'left',
                                 'full', 'grid', 'lowres'],
                        default='center')
    parser.add_argument('--centerScale', type=float, default=0.25)
    parser.add_argument('imgs', type=str, nargs='+')

    hps = parser.parse_args()  # So error if typo

    main(hps)
