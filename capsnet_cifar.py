from __future__ import absolute_import

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import random
import time
import sys
import copy
import uuid
import cPickle as pickle
import numpy as np
# import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

import tensorflow as tf


# from tensorflow.python import debug as tf_debug
# from tensorflow.python.ops import math_ops, variable_scope, tensor_array_ops
# from tensorflow.contrib.tensorboard.plugins import projector

def unpickle(file):
	fo = open(file, 'rb')
	dict = pickle.load(fo)
	fo.close()
	return dict['data'], dict['labels']


def cifar_generator(filenames, batch_size, data_dir):
	all_data = []
	all_labels = []
	for filename in filenames:
		data, label = unpickle(data_dir + '/' + filename)
		all_data.append(data)
		all_labels.append(label)

	images = np.concatenate(all_data, axis=0)
	# images = (images.astype(np.float32)/127.5) - 1
	images = images.astype(np.float32) / 255.
	images = np.reshape(images, (-1, 3, 32, 32))
	images = np.transpose(images, (0, 2, 3, 1))

	labels = np.concatenate(all_labels, axis=0)

	batched = list(zip(images, labels))

	def get_epoch():
		# remove randomness below to compare diff runs for params search
		# put it back if averaging multiple runs for each param value

		# np.random.shuffle(batched)
		images, labels = zip(*batched)
		for i in xrange(len(images) / batch_size):
			image_batch = np.copy(images[i * batch_size:(i + 1) * batch_size])
			label_batch = np.copy(labels[i * batch_size:(i + 1) * batch_size])
			yield (image_batch, label_batch)

	return get_epoch


def generate_cifar(batch_size, data_dir):
	return (
		cifar_generator(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'],
						batch_size,
						data_dir),
		cifar_generator(['test_batch'], batch_size, data_dir)
	)


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME',
		   bias=True, name="conv2d", init='he', verbose=False):
	with tf.variable_scope(name):
		if init == "he":
			shape = input_.get_shape().as_list()
			fan_in = shape[-1] * k_h * k_w
			fan_out = output_dim * k_h * k_w / (d_h * d_w)
			stddev = np.sqrt(4. / (fan_in + fan_out))
		elif init == "he_per_CS231n":  # http://cs231n.github.io/neural-networks-2/#init
			# https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/ops/init_ops.py
			shape = input_.get_shape().as_list()
			fan_in = shape[-1] * k_h * k_w
			stddev = np.sqrt(2. / fan_in)

		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
		if verbose:
			print
			conv
		if bias:
			biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		else:
			biases = tf.zeros([output_dim])

		conv = tf.nn.bias_add(conv, biases)

		return conv


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, init='he', with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		if init == "he":
			dim = tf.shape(input_)[1]
			stddev = tf.sqrt(2.0 / tf.cast(dim, dtype=tf.float32))

		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
							   initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias


def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak * x)


def print_grads_and_vars(opt, loss, var_list, print_names=True):
	grads_and_vars = opt.compute_gradients(loss, var_list)
	gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
	if print_names:
		for g in gradients:
			print
			g
		for v in variables:
			print
			v
		print
		"===========\n\n"
	return gradients


def preprocess_input(batch, shift=True, pad=0, noise=[0, 0], cutout=[0, 0, 0, 0, 0], random=True):
	if shift:
		batch = batch * 2 - 1
	if len(batch.shape) == 3:
		batch = np.expand_dims(batch, axis=-1)
	if pad:
		batch = np.lib.pad(batch, ((0, 0), (0, 0), (pad, pad), (0, 0)), 'constant')

	if cutout[2]:
		# [window time dim, window freq dim, number of windows, value to set to (0 or -1 for music samples), probability to insert each window]
		# np.set_printoptions(formatter={'float': '{: 0.2f}'.format}, linewidth=175)
		# randomly remove chunks of the input (batch_size, time span, freq span, channels)
		window_height, window_width, num_windows, value, prob = cutout
		# value should be -1 if shift, 1 otherwise:
		if shift and value != -1:
			print
			"\n\n\n\n*******  Attention: cutout is inserting {} instead of -1  *******\n\n\n\n".format(value)
		if not shift and value != 0:
			print
			"\n\n\n\n*******  Attention: cutout is inserting {} instead of 0  *******\n\n\n\n".format(value)
		batch_size = batch.shape[0]
		input_height = batch.shape[1]  # time
		input_width = batch.shape[2]  # frequency

		if random:
			for i in range(num_windows):
				# pick random locations of the windows' top left pixels:
				x = np.random.randint(input_height - window_height, size=batch_size)  # time
				y = np.random.randint(pad, input_width - window_width - pad, size=batch_size)  # freq
				samples = np.random.randint(100, size=batch_size)
				for img, s in zip(range(batch_size), samples):
					if s < prob * 100:
						batch[img, x[img]:x[img] + window_height, y[img]:y[img] + window_width, :] = value
		else:
			# less random: fill same locations throughout the batch:
			for i in range(num_windows):
				x = np.random.randint(input_height - window_height)
				y = np.random.randint(pad, input_width - window_width - pad)
				batch[:, x:x + window_height, y:y + window_width, :] = value

	if noise[0] or noise[1]:
		# [remove zeros, remove ones]
		mask = np.random.binomial(1, noise[0], size=batch.shape).astype(np.bool)
		batch[mask] = cutout[3]
		mask = np.random.binomial(1, noise[1], size=batch.shape).astype(np.bool)
		batch[mask] = 1

	return batch


"""
class CapsNet(object):
	def __init__(self, batch_size, num_fmaps, num_pcaps, num_classes, cap_sizes, num_iterations, filters=[9, 9],
				 strides=[1, 2], scope="capsnet"):
		self.batch_size = batch_size
		self.num_fmaps = num_fmaps
		self.num_pcaps = num_pcaps
		self.num_classes = num_classes
		self.cap_sizes = cap_sizes
		self.num_iterations = num_iterations
		self.scope = scope
		self.filters = filters
		self.strides = strides

	def __call__(self, images, reuse=False, train=True):
		caps = []
		with tf.variable_scope(self.scope, reuse=reuse):
			input_shape = images.get_shape().as_list()
			print 'Processing input images {} with filters=({:d},{:d}) , stride={:d}:\n'.format(
				input_shape, self.filters[0], self.filters[0], self.strides[0])
			conv_out = conv2d(images, self.num_fmaps, k_h=self.filters[0], k_w=self.filters[0], d_h=self.strides[0],
							  d_w=self.strides[0],
							  padding="SAME", name='conv_layer_' + str(self.num_fmaps))
			conv_out = lrelu(conv_out, name='relu_conv_layer_' + str(self.num_fmaps))

			# initial PrimaryCaps outputs:
			# TODO: move into CapsLayer function to allow multiple convolutional capsule layers
			for i in range(self.num_pcaps):
				cap = conv2d(conv_out, self.cap_sizes[0], k_h=self.filters[1], k_w=self.filters[1], d_h=self.strides[1],
							 d_w=self.strides[1],
							 padding="SAME", name='pcap_layer' + str(i))  # (128,6,6,8)
				caps.append(cap)

			S = tf.concat(caps, 0)  # (32, 128, 6, 6, 8)
			S = tf.transpose(S, perm=(1, 0, 2, 3, 4))  # make batch_size leading dim
			V = tf.map_fn(S, self.squash)  # applies function to last dim, reshape before or after?
			V = tf.reshape(V, [self.batch_size, -1, V.shape[-1]])  # (128,1152,8)
			dcaps = self.CapsLayer(V, self.cap_sizes[1], self.num_classes,
								   num_iterations=self.num_iterations)  # (10,16)

		return dcaps

	def squash(S):
		return tf.norm(S) * S / (1 + tf.square(tf.norm(S)))

	def CapsLayer(self, V, output_cap_size, output_num_caps, num_iterations=3, name="CapsLayer"):
		_, input_num_caps, input_cap_size = V.get_shape().as_list()

		with tf.variable_scope(name):
			# weights between PrimaryCaps and DigitCaps
			W = tf.get_variable('w_intercaps', [output_num_caps, input_num_caps, input_cap_size, output_cap_size],
								initializer=tf.truncated_normal_initializer(stddev=0.01))  # (10,1152,8,16)
			# Einsum op: #http://ajcr.net/Basic-guide-to-einsum/
			# repeating letter in both input arrays is the dim to multiply along, missing letter in output array is the dim to sum along
			# all pcap vectors connect to all dcap vectors using transformation matrices W for each connection
			U = tf.einsum('bjk,ijkl->bijl', V,
						  W)  # (128, 1152, 8)x(10, 1152, 8, 16) -> (128, 10, 1152, 16)  prediction vectors from each pcapsule to each dcapsule

			# zero out log priors (dynamic connection weights (do not confuse with W, which is transformation weights)):
			B = tf.zeros((self.batch_size, input_num_caps, output_num_caps))  # (128, 1152, 10)

			# routing algorithm
			# for each forward pass, find connection weights which maximize capsules agreement (cosine distance between outputs):
			for r in num_iterations:
				C = tf.softmax(B)  # c_IJ = exp(b_IJ) / sum_K(exp(b_IK)), K=10
				S = tf.einsum('bji,bijk->bik', C,
							  U)  # S = tf.reduce_sum(C*U, axis=1)  (128, 1152, 10)*(128, 10, 1152, 16) -> (128, 10, 16)
				V = tf.map_fn(S, self.squash)
				B += tf.einsum('bijk,bik->bji', U,
							   V)  # tf.dot(U, V)   #(128, 10, 1152, 16).(128, 10,16) -> (128, 1152,10)

			return V  # (128,10,16)

"""


class CapsNet(object):
	def __init__(self, batch_size, num_fmaps, num_pcaps, num_classes, cap_sizes, num_iterations, filters=[9, 9],
				 strides=[1, 2], scope="capsnet"):
		self.batch_size = batch_size
		self.num_fmaps = num_fmaps
		self.num_pcaps = num_pcaps
		self.num_classes = num_classes
		self.cap_sizes = cap_sizes
		self.num_iterations = num_iterations
		self.scope = scope
		self.filters = filters
		self.strides = strides

	def __call__(self, images, labels, reuse=False, train=True):
		caps = []
		with tf.variable_scope(self.scope, reuse=reuse):
			input_shape = images.get_shape().as_list()
			print
			'Processing input images {} with filters=({:d},{:d}) , stride={:d}:\n'.format(
				input_shape, self.filters[0], self.filters[0], self.strides[0])
			conv_out = conv2d(images, self.num_fmaps, k_h=self.filters[0], k_w=self.filters[0], d_h=self.strides[0],
							  d_w=self.strides[0],
							  padding="VALID", name='conv_layer_' + str(self.num_fmaps))
			conv_out = lrelu(conv_out, name='relu_conv_layer_' + str(self.num_fmaps))
			print
			conv_out.shape

			# initial PrimaryCaps outputs:
			# TODO: move into CapsLayer function to allow multiple convolutional capsule layers
			for i in range(self.num_pcaps):
				cap = conv2d(conv_out, self.cap_sizes[0], k_h=self.filters[1], k_w=self.filters[1], d_h=self.strides[1],
							 d_w=self.strides[1],
							 padding="SAME", name='pcap_layer' + str(i))  # (128,6,6,8)
				# print cap.shape,
				caps.append(cap)

			# S = tf.concat(caps, 0)   #(32, 128, 6, 6, 8)
			S = tf.convert_to_tensor(caps, dtype=tf.float32)
			print
			S.shape
			S = tf.transpose(S, perm=(1, 0, 2, 3, 4))  # make batch_size leading dim
			print
			S.shape
			V = tf.map_fn(self.squash, S)  # applies function to last dim, reshape before or after?
			V = tf.reshape(V, [self.batch_size, -1, V.shape[-1]])  # (128,1152,8)
			print
			V.shape
			dcaps = self.caps_layer(V, self.cap_sizes[1], self.num_classes,
									num_iterations=self.num_iterations)  # (128,10,16)
			print
			dcaps.shape
			"""
			array = tf.Variable(tf.zeros((b, n, m)))
			ones = tf.Variable(tf.ones((m)))
			labels = tf.Variable(tf.ones((b), dtype=tf.int32))

			sess = tf.Session()
			with sess.as_default():
				sess.run(tf.global_variables_initializer())
				for i in range(b):
					with tf.control_dependencies([array[i:labels[i]].assign(
							tf.ones(m))]):  # changed [array[i][labels[i]] --->  [array[i:labels[i]]
						print(array)  # removed tf.identity(array)

				array = tf.identity(array)
				print array.shape

			"""

			# output for the reconstructing decoder - zero out outputs from all capsules other than the one that should be correct
			mask = tf.Variable(tf.zeros(dcaps.shape), trainable=False, name="mask")
			ones = tf.Variable(tf.ones(dcaps.shape[-1]), trainable=False, name="ones")
			labels = tf.Variable(tf.ones(dcaps.shape[0], dtype=tf.int32), trainable=False)
			print
			mask.shape
			print
			ones.shape
			print
			labels.shape
			print
			"""
			b = labels[0]
			sess = tf.Session()
			with sess.as_default():
				sess.run(tf.global_variables_initializer())
				print sess.run(mask[0][0])
				mask[0][0].eval()
				print mask.shape
				print sess.run(ones)
				mask[0,0].assign(ones)
				print sess.run(mask[0,0])
				result = mask[0, 0].assign(ones)
				print sess.run(result[0,0])

			print "\n\ntest\n\n"

			#mask[0][0].assign(ones)
			"""
			# for i in range(self.batch_size):
			# mask[i][self.class_labels[i]] = 1  set the output from the capsule at the correct position to ones
			with tf.control_dependencies([mask[i, labels[i]].assign(ones) for i in range(self.batch_size)]):
				mask = tf.identity(mask)  # make sure it gets executed
			# print mask.shape,
			print
			print
			mask.shape
			print
			# """
			# insert ones at locations specified by indices into zero tensor of shape dcaps:  scatter_nd(indices, updates, shape)
			# for each of the elements in updates we need 3 indices to determine where in the output each element is going to be placed
			# The last dimension of indices can be at most the rank of shape (if less than rank, than it will
			# TODO: mask = tf.scatter_nd(labels, tf.ones((self.batch_size, self.cap_sizes[1])), dcaps.shape)

			dcaps_correct = dcaps * mask

		return dcaps, dcaps_correct

	@staticmethod
	def squash(S):
		return tf.norm(S) * S / (1 + tf.square(tf.norm(S)))

	def caps_layer(self, V, output_cap_size, output_num_caps, num_iterations=3, name="CapsLayer"):
		_, input_num_caps, input_cap_size = V.get_shape().as_list()

		with tf.variable_scope(name):
			# weights between PrimaryCaps and DigitCaps
			W = tf.get_variable('w_intercaps', [output_num_caps, input_num_caps, input_cap_size, output_cap_size],
								initializer=tf.truncated_normal_initializer(stddev=0.01))  # (10,1152,8,16)
			# Einsum op: #http://ajcr.net/Basic-guide-to-einsum/
			# repeating letter in both input arrays is the dim to multiply along, missing letter in output array is the dim to sum along
			# all pcap vectors connect to all dcap vectors using transformation matrices W for each connection
			U = tf.einsum('bjk,ijkl->bijl', V,
						  W)  # (128, 1152, 8)x(10, 1152, 8, 16) -> (128, 10, 1152, 16)  prediction vectors from each pcapsule to each dcapsule

			# zero out log priors (dynamic connection weights (do not confuse with W, which is transformation weights)):
			B = tf.zeros((self.batch_size, input_num_caps, output_num_caps))  # (128, 1152, 10)

			# routing algorithm
			# for each forward pass, find connection weights which maximize capsules agreement (cosine distance between outputs):
			for r in range(num_iterations):
				C = tf.nn.softmax(B)  # c_IJ = exp(b_IJ) / sum_K(exp(b_IK)), K=10
				S = tf.einsum('bji,bijk->bik', C,
							  U)  # S = tf.reduce_sum(C*U, axis=1)  (128, 1152, 10)*(128, 10, 1152, 16) -> (128, 10, 16)
				V = tf.map_fn(self.squash, S)
				B += tf.einsum('bijk,bik->bji', U,
							   V)  # tf.dot(U, V)   #(128, 10, 1152, 16).(128, 10,16) -> (128, 1152,10)

			return V  # (128,10,16)


class CapsDecoder(object):
	# input shape (batch_size, num_classes, dcap length)
	# output shape (batch_size, output_dim, output_dim, channels)
	# (self.batch_size, (256, 256), (32, 32), self.channels, scope='decoder')

	def __init__(self, batch_size, layers, output_dims, channels=3, scope='decoder'):
		self.batch_size = batch_size
		self.layers = layers
		self.out_channels = channels
		self.scope = scope
		self.output_dims = output_dims

	def __call__(self, features, dropout=0.5, reuse=False, train=True):
		with tf.variable_scope(self.scope, reuse=reuse):
			print
			'\nProcessing input features {}:\n'.format(features.shape)
			dim_x, dim_y = self.output_dims

			x = tf.reshape(features, (self.batch_size, -1))

			for i, h in enumerate(self.layers + (dim_x * dim_y * self.out_channels,)):
				# print '\n\nh:', str(h), 'x.shape:', x.shape, '\n\n'
				print
				'decoder output layer: {} by {}'.format(x.get_shape()[1], h)
				x = linear(x, h, 'decoder_hidden_layer' + str(i) + '_' + str(h))
				# print '_decoder output layer: {} by {}'.format(h, x.get_shape()[1])
				if i == len(self.layers):
					x = tf.nn.sigmoid(x, name='sigmoid_output_image')
				else:
					x = lrelu(x, name='relu_decoder_' + str(i) + '_' + str(h))

			image_reconstr = tf.reshape(x, (self.batch_size, dim_x, dim_y, self.out_channels))
			print
			"\nReshaping output vector to", image_reconstr.shape

			return image_reconstr


class Model(object):

	def __init__(self):
		self.debug = False
		self.check_gradients = False
		self.batch_size = 128
		self.model_epochs = 4
		self.model_LR = 0.0005
		self.model_dropout = 0.5
		self.cutout = [20, 20, 8, -1, 0.5]
		self.noise = [0.05, 0.005]
		self.shift_input = False  # if true shift input from [0,1] to [-1,1] range
		self.model_beta1 = 0.9  # for ADAM optimizer in the autoencoder
		self.ae_cost_type = "mse"
		self.deep_classifier_cost_type = "ce"
		self.feat_act = 'identity'
		self.ae_weight = 10

		self.num_classes = 10
		self.cifar_path = '/shared/UsedDatabase/' + 'cifar10.pkl'
		print
		"\nLoading CIFAR generator...\n"
		import time
		start = time.time()
		self.train_gen, self.test_gen = generate_cifar(self.batch_size,
													   data_dir="cifar-10-batches-py")  # /shared/UsedDatabase/cifar-10-batches-py")
		print
		"Done. Took {:.2f} sec\n".format(time.time() - start)
		self.channels = 3
		self.num_train_batches = 50000 / self.batch_size
		self.num_test_batches = 10000 / self.batch_size

		self.graph = tf.Graph()
		with self.graph.as_default():
			self.setup_placeholders()
			self.build_model()
			self.build_model_loss()
			self.build_model_training()
			self.model_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	def setup_placeholders(self):
		self.dropout = tf.placeholder(tf.float32, name='dropout')
		self.orig_input = tf.placeholder(tf.float32,
										 shape=[self.batch_size, 32, 32, self.channels],
										 name='orig_input_images')
		self.encoder_input = tf.placeholder(tf.float32,
											shape=[self.batch_size, 32, 32, self.channels],
											name='corrupted_input_images')

		self.class_labels = tf.placeholder(tf.int32, shape=[None], name='labels')

	def build_model(self):
		print
		"\n\nBuilding {} model:\n\n".format("CapsulesNet")
		print
		"Building encoder ({})\n".format("CapsNet")
		# batch_size, num_fmaps, num_pcaps, num_classes, cap_sizes, num_iterations, filters=[9,9], strides=[1,2], scope="capsnet"
		self.encoder = CapsNet(self.batch_size, 16, 32, 10, [8, 16], 3)
		# images, labels, reuse=False, train=True
		self.features, correct_features = self.encoder(self.encoder_input, self.class_labels)

		self.enc_vars = [var for var in tf.global_variables() if var.name.startswith('encoder')]
		self.enc_weights = [var for var in self.enc_vars if '/w' in var.name]
		self.enc_biases = [var for var in self.enc_vars if '/bias' in var.name]

		print
		"\nBuilding reconstructing decoder ({})".format("CapsDecoder")
		self.decoder = CapsDecoder(self.batch_size, (256, 256), (32, 32), self.channels, scope='decoder')

		"""
		# zero out all but the correct capsule activity: features (128,10,16), class_labels: (128)
		correct_features = tf.zeros(self.features.shape)
		# TODO vectorize the following code, perhaps with tf.gather_nd()
		for i in range(self.batch_size):
			for j in range(self.num_classes):
				correct_features[i, j] = self.features[i, j]
		# TODO connect graphs, so that self.features are used both as input to classifier, and input to AE.
		"""

		self.reconstructed = self.decoder(correct_features)

		self.dec_vars = [var for var in tf.global_variables() if var.name.startswith('decoder')]
		self.dec_weights = [var for var in self.dec_vars if '/w' in var.name]
		self.dec_biases = [var for var in self.dec_vars if '/bias' in var.name]

		self.ae_vars = self.enc_vars + self.dec_vars

		self.model_vars = [var for var in tf.trainable_variables() if ('coder' in var.name or "classifier" in var.name)]
		self.model_weights = [var for var in self.model_vars if ('/w' in var.name or 'Matrix' in var.name)]
		self.model_biases = [var for var in self.model_vars if '/bias' in var.name]
		self.model_bn_vars = [var for var in self.model_vars if '/bn' in var.name]

	def build_deep_classifier_loss(self):
		with tf.variable_scope("capsnet_loss") as scope:
			self.deep_classifier_loss = 0

			norms = tf.norm(self.features, axis=1)  # (128, 10, 16) -> (128, 10)  labels: (128)

			for i in range(self.num_classes):
				presence = tf.cast(tf.equal(self.class_labels, i),
								   tf.float32)  # binary vector indicating if this output position should contain the correct label
				loss = presence * tf.square(tf.maximum(0., 0.9 - norms[:, i]) + 0.5 * (1. - presence) * tf.square(
					tf.maximum(0., norms[:, i] - 0.1)))  # (128)
				avg_loss = tf.reduce_mean(loss)
				self.deep_classifier_loss += avg_loss

			# predictions = tf.argmax(norms, axis=1)
			predictions = tf.argmax(norms, axis=1, output_type=tf.int32)
			correct = tf.cast(tf.equal(predictions, self.class_labels), tf.float32)
			self.deep_classifier_accuracy = tf.reduce_mean(correct, name='deep_classifier_mean_of_correct')

	def build_autoencoder_loss(self):
		with tf.variable_scope("autoencoder_loss") as scope:
			if self.ae_cost_type == 'mse':
				print
				"\n\nUsing MSE loss\n\n"
				self.ae_loss = tf.reduce_mean(tf.square(self.orig_input - self.reconstructed))

			elif self.ae_cost_type == 'ce':
				print
				"\n\nUsing Cross-Entropy loss\n\n"
				self.ae_loss = -tf.reduce_mean(self.orig_input * tf.log(self.reconstructed + 0.0000001) + \
											   (1 - self.orig_input) * tf.log(1 - self.reconstructed + 0.0000001))

			elif self.ae_cost_type == 'nll':
				print
				"\n\nUsing NLL loss\n\n"
				self.ae_loss = -tf.reduce_mean(self.orig_input * tf.log(self.reconstructed + 0.0000001))

	def build_model_loss(self):
		self.build_deep_classifier_loss()
		self.build_autoencoder_loss()
		self.model_loss = self.deep_classifier_loss + self.ae_weight * self.ae_loss

	def build_model_training(self):
		with tf.variable_scope("model_training"):
			optimizer = tf.train.AdamOptimizer(self.model_LR, beta1=self.model_beta1)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.model_train_op = optimizer.minimize(self.model_loss)  # , var_list=self.ae_vars)

			if self.debug:
				print_names = True
				print
				"\n\nModel variables and gradients:\n"
			else:
				print_names = False

			self.model_grads = print_grads_and_vars(optimizer, self.model_loss, tf.trainable_variables(),
													print_names=print_names)
			# print "\nprinting self.model_grads:"
			# for var in self.model_grads:
			# print var.name
			# print "\nDone\n"
			self.model_w_grads = [var for var in self.model_grads if 'Conv2D_grad' in var.name
								  or 'conv2d_transpose_grad' in var.name or 'MatMul_grad' in var.name]
			self.model_b_grads = [var for var in self.model_grads if 'Bias' in var.name or 'add_grad' in var.name]
			self.model_bn_grads = [var for var in self.model_grads if 'batchnorm' in var.name]

	def check_test_loss(self, sess, num_batches):
		batch_losses = []
		batch_number = 0
		test_cifar = self.inf_test_gen()

		while batch_number < num_batches:
			test_images, test_labels = test_cifar.next()
			orig_test_images = np.copy(test_images)
			corrupted_test_images = preprocess_input(test_images,
													 shift=self.shift_input)  # , noise=self.noise, cutout=self.cutout)

			total_loss, dcl_loss, ae_loss = sess.run([
				self.model_loss, self.deep_classifier_loss, self.ae_loss],
				feed_dict={self.encoder_input: corrupted_test_images,
						   self.orig_input: orig_test_images, self.class_labels: test_labels, self.dropout: 1.0})
			batch_losses.append([total_loss, dcl_loss, ae_loss])

			batch_number += 1

		return np.mean(batch_losses, axis=0)

	def check_test_accuracy(self, sess):
		batch_number = 0
		batch_accuracies = []

		test_cifar = self.inf_test_gen()

		# test_input is features for linear classifier, images for deep classifier
		while batch_number < self.num_test_batches:
			test_input, test_labels = test_cifar.next()

			accuracy = sess.run(self.deep_classifier_accuracy, feed_dict={self.encoder_input: test_input,
																		  self.dropout: 1.0,
																		  self.class_labels: test_labels})

			batch_accuracies.append(accuracy)
			batch_number += 1

		return np.mean(batch_accuracies)

	def inf_train_gen(self):
		# this is for Cifar-10
		while True:
			for images, labels in self.train_gen():
				yield images, labels

	def inf_test_gen(self):
		while True:
			for images, labels in self.test_gen():
				yield images, labels

	def train(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		with tf.Session(config=config, graph=self.graph) as sess:

			# self.writer = tf.summary.FileWriter(self.log_dir, sess.graph)
			sess.run(self.model_init_op)

			if self.debug:
				print
				"\n\n************ Initialization Values ***************\n\n"
				print_vars_and_gradients(self, sess, only_weights=True, outputs=False)

			train_set, test_set = [], []
			train_cifar = self.inf_train_gen()
			test_cifar = self.inf_test_gen()
			# save the first batch for viewing purposes
			train_images, self.saved_train_labels = train_cifar.next()
			test_images, self.saved_test_labels = test_cifar.next()

			print
			"\n\nTraining model for {:d} epochs, LR = {:.4f}, {} training batches, {} test batches\n".format(
				self.model_epochs, self.model_LR, self.num_train_batches, self.num_test_batches)

			epoch = 0
			iteration = 0
			best_epoch = 0
			train_loss = 0
			train_accuracy = 0
			test_accuracy = 0

			while epoch < self.model_epochs:

				train_batch_number = 0
				train_batch_losses = []
				train_batch_accuracies = []

				while train_batch_number < self.num_train_batches:
					train_images, train_labels = train_cifar.next()
					orig_train_images = np.copy(train_images)
					# passing train_images into process_input as argument will corrupt them (it's a pointer!)
					corrupted_train_images = preprocess_input(train_images,
															  shift=self.shift_input)  # , noise=self.noise, cutout=self.cutout)

					if self.debug and train_batch_number == 0:
						np.set_printoptions(formatter={'float': '{:3.1f}'.format}, linewidth=150)
						print
						"\n\norig_train_images", orig_train_images.shape, '\n\n', orig_train_images[0, :, :, 0]
						print
						"\n\ncorrupted_train_images", corrupted_train_images.shape, '\n\n', corrupted_train_images[0, :,
																							:, 0]
						print
						"\n\ntrain_labels:", train_labels.shape, train_labels[:20]
						print
						"\n\nPlotting Image Reconstructions...\n"
						plot_image_reconstructions(self, sess, orig_train_images, train_labels, orig_train_images,
												   train_labels, corrupted_train_images, corrupted_train_images,
												   path='/ssd-2/michael/capsnet/plots/',
												   tag='before_training.png', midi=False)
						print
						"\nDone!\n\n"

					# if self.debug:
					# print "\n\n\n\n********************** Before processing the first batch: **********************\n\n\n\n"
					# print_vars_and_gradients(self, sess, orig_train_images, train_labels, vars_and_grads=(
					# self.model_weights, self.model_w_grads, self.model_biases, self.model_b_grads),
					# batchnorm=(self.model_bn_vars, self.model_bn_grads))
					_, train_loss, train_accuracy = sess.run(
						[self.model_train_op, self.model_loss, self.deep_classifier_accuracy],
						feed_dict={self.encoder_input: corrupted_train_images, self.dropout: self.model_dropout,
								   self.orig_input: orig_train_images, self.class_labels: train_labels})

					if self.debug and train_batch_number % 5 == 0 and train_batch_number < 200:
						# print "\n\n\n\n********************** After processing the first batch: ************************\n\n\n\n"
						print
						"\n\nBatch number:", train_batch_number, '\n\n'
						print
						"\n\ntrain_loss: {:.5f}, train_accuracy: {:.4f}\n\n\n\n".format(train_loss,
																						train_accuracy)
						# print_vars_and_gradients(self, sess, orig_train_images, train_labels, vars_and_grads=(
						# self.model_weights, self.model_w_grads, self.model_biases, self.model_b_grads),
						# batchnorm=(self.model_bn_vars, self.model_bn_grads))
						print
						"\n\nPlotting Image Reconstructions...\n"
						plot_image_reconstructions(self, sess, orig_train_images, train_labels, orig_train_images,
												   train_labels, corrupted_train_images, corrupted_train_images,
												   path='/ssd-2/michael/capsnet/plots/',
												   tag='before_training.png', midi=False)
						print
						"\nDone!\n\n"

					# for long epochs, print out progress every 1/10th epoch:
					# if train_batch_number % int(self.num_train_batches / 10) == 0:
					# print "batch {:d}: {}".format(train_batch_number, str(datetime.now())[11:-7])

					train_batch_losses.append(train_loss)  # will be list of zeros for classifier unless debugging
					train_batch_accuracies.append(train_accuracy)  # will be list of zeros for autoencoder
					train_batch_number += 1

					if self.check_gradients:
						check_gradients_error(sess, self.model_vars, self.model_loss,
											  extra_feed_dict={self.encoder_input: corrupted_train_images,
															   self.dropout: 1.0,
															   self.orig_input: orig_train_images,
															   self.class_labels: train_labels})

				train_batch_losses.append(train_loss)  # will be list of zeros for classifier
				train_batch_accuracies.append(train_accuracy)  # will be list of zeros for autoencoder

				# for long epochs, print out progress every 1/10th epoch:
				# if train_batch_number % int(self.num_train_batches / 10) == 0:
				# print "batch {:d}: {}".format(train_batch_number, str(datetime.now())[11:-7])
				# train_batch_number += 1

				if self.check_gradients:
					check_gradients_error(sess, self.model_vars, self.model_loss,
										  extra_feed_dict={self.encoder_input: corrupted_train_images,
														   self.dropout: 1.0,
														   self.orig_input: orig_train_images,
														   self.class_labels: train_labels})

				# for each epoch, display and record train and test results:
				avg_train_loss = np.mean(train_batch_losses)
				avg_train_accuracy = np.mean(train_batch_accuracies)
				avg_test_accuracy = self.check_test_accuracy(sess)
				total_test_loss, dcl_loss, avg_test_ae_loss = self.check_test_loss(sess, self.num_test_batches)

				print(
					"Epoch {:d}/{:d}: {} | Loss: train {:.5f}, test {:.5f} (cl {:.3f} ae {:.3f}) | Accuracy: train {:.3f} test {:.3f})".format(
						epoch + 1, self.model_epochs, str(datetime.now())[11:-7], avg_train_loss, total_test_loss,
						dcl_loss,
						self.ae_weight * avg_test_ae_loss, avg_train_accuracy, avg_test_accuracy))

				# increment epoch number here to use % operator when epoch=0
				epoch += 1
		sess.close()


CapsulesNet = Model()
CapsulesNet.train()
