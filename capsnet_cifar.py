import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime
import numpy as np
import tensorflow as tf

from utils import *

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
			print 'Processing input images {} with filters=({:d},{:d}) , stride={:d}:\n'.format(
				input_shape, self.filters[0], self.filters[0], self.strides[0])
			conv_out = conv2d(images, self.num_fmaps, k_h=self.filters[0], k_w=self.filters[0], d_h=self.strides[0],
							  d_w=self.strides[0], padding="VALID", name='conv_layer_' + str(self.num_fmaps))
			conv_out = lrelu(conv_out, name='relu_conv_layer_' + str(self.num_fmaps))
			print 'conv_out:', conv_out.shape

			# initial PrimaryCaps outputs:
			# TODO: move into CapsLayer function to allow multiple convolutional capsule layers
			for i in range(self.num_pcaps):
				cap = conv2d(conv_out, self.cap_sizes[0], k_h=self.filters[1], k_w=self.filters[1], d_h=self.strides[1],
							 d_w=self.strides[1], padding="SAME", name='pcap_layer' + str(i))  # (128,6,6,8)
				caps.append(cap)

			S = tf.convert_to_tensor(caps, dtype=tf.float32)
			print 'S:', S.shape
			S = tf.transpose(S, perm=(1, 0, 2, 3, 4))  # make batch_size leading dim
			print 'S after transpose:', S.shape
			V = tf.map_fn(self.squash, S)  # applies function to last dim, reshape before or after?
			V = tf.reshape(V, [self.batch_size, -1, V.shape[-1]])  # (128,1152,8)
			print 'V:', V.shape
			dcaps = self.caps_layer(V, self.cap_sizes[1], self.num_classes,
									num_iterations=self.num_iterations)  # (128,10,16)
			print 'dcaps:', dcaps.shape

			# output for the reconstructing decoder - zero out outputs from all capsules other than the one that should be correct
			mask = tf.Variable(tf.zeros(dcaps.shape), trainable=False, name="mask")
			ones = tf.Variable(tf.ones(dcaps.shape[-1]), trainable=False, name="ones")
			labels = tf.Variable(tf.ones(dcaps.shape[0], dtype=tf.int32), trainable=False)

			# for i in range(self.batch_size):
			# mask[i][self.class_labels[i]] = 1  set the output from the capsule at the correct position to ones
			with tf.control_dependencies([mask[i, labels[i]].assign(ones) for i in range(self.batch_size)]):
				mask = tf.identity(mask)  # make sure it gets executed

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
			U = tf.einsum('bjk,ijkl->bijl', V, W) # (128, 1152, 8)x(10, 1152, 8, 16) -> (128, 10, 1152, 16)

			# zero out log priors (dynamic connection weights (do not confuse with W, which is transformation weights)):
			B = tf.zeros((self.batch_size, input_num_caps, output_num_caps))  # (128, 1152, 10)

			# routing algorithm
			# for each forward pass, find connection weights which maximize capsules agreement (cosine distance between outputs):
			for r in range(num_iterations):
				C = tf.nn.softmax(B)  # c_IJ = exp(b_IJ) / sum_K(exp(b_IK)), K=10
				S = tf.einsum('bji,bijk->bik', C, U)  # S = tf.reduce_sum(C*U, axis=1)  (128, 1152, 10)*(128, 10, 1152, 16) -> (128, 10, 16)
				V = tf.map_fn(self.squash, S)
				B += tf.einsum('bijk,bik->bji', U, V)  # tf.dot(U, V)   #(128, 10, 1152, 16).(128, 10,16) -> (128, 1152,10)

			return V  # (128,10,16)


class CapsDecoder(object):
	# input shape (batch_size, num_classes, dcap length)
	# output shape (batch_size, output_dim, output_dim, channels)

	def __init__(self, batch_size, layers, output_dims, channels=3, scope='decoder'):
		self.batch_size = batch_size
		self.layers = layers
		self.out_channels = channels
		self.scope = scope
		self.output_dims = output_dims

	def __call__(self, features, reuse=False, train=True):
		with tf.variable_scope(self.scope, reuse=reuse):
			print '\nProcessing input features {}:\n'.format(features.shape)
			dim_x, dim_y = self.output_dims

			x = tf.reshape(features, (self.batch_size, -1))

			for i, h in enumerate(self.layers + (dim_x * dim_y * self.out_channels,)):
				print 'decoder output layer: {} by {}'.format(x.get_shape()[1], h)
				x = linear(x, h, 'decoder_hidden_layer' + str(i) + '_' + str(h))

				if i == len(self.layers):
					x = tf.nn.sigmoid(x, name='sigmoid_output_image')
				else:
					x = lrelu(x, name='relu_decoder_' + str(i) + '_' + str(h))

			image_reconstr = tf.reshape(x, (self.batch_size, dim_x, dim_y, self.out_channels))
			print "\nReshaping output vector to", image_reconstr.shape

			return image_reconstr


class Model(object):

	def __init__(self, LR=0.001, batch_size=128, epochs=40, ae_cost_type='mse', ae_weight=10, num_classes=10,
				 channels=3, img_dim=32, beta1=0.9, debug=False):
		self.batch_size = batch_size
		self.epochs = epochs
		self.LR = LR
		self.beta1 = beta1  # for ADAM optimizer in the autoencoder
		self.ae_cost_type = ae_cost_type
		self.ae_weight = ae_weight
		self.num_classes = num_classes
		self.debug = debug
		self.channels = channels
		self.img_dim = img_dim
		self.num_train_batches = 50000 / self.batch_size
		self.num_test_batches = 10000 / self.batch_size

		print "\nLoading CIFAR generator...\n"
		self.train_gen, self.test_gen = generate_cifar(self.batch_size, data_dir="cifar-10-batches-py")

		self.graph = tf.Graph()
		with self.graph.as_default():
			self.setup_placeholders()
			self.build_model()
			self.build_model_loss()
			self.build_model_training()
			self.model_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

	def setup_placeholders(self):
		self.orig_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_dim, self.img_dim, self.channels], name='orig_input_images')
		self.encoder_input = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_dim, self.img_dim, self.channels], name='encoder_input_images')
		self.class_labels = tf.placeholder(tf.int32, shape=[None], name='labels')

	def build_model(self):
		print "\nBuilding Encoder ({})\n".format("CapsNet")
		# batch_size, num_fmaps, num_pcaps, num_classes, cap_sizes, num_iterations, filters=[9,9], strides=[1,2], scope="capsnet"
		self.encoder = CapsNet(self.batch_size, 64, 32, self.num_classes, [8, 16], 3)
		self.features, correct_features = self.encoder(self.encoder_input, self.class_labels)

		print "\nBuilding reconstructing decoder ({})".format("CapsDecoder")
		self.decoder = CapsDecoder(self.batch_size, (256, 256), (self.img_dim, self.img_dim), self.channels, scope='decoder')
		self.reconstructed = self.decoder(correct_features)


	def build_classifier_loss(self):
		print "\nUsing margin loss for capsnet classifier"
		with tf.variable_scope("capsnet_loss") as scope:
			self.classifier_loss = 0

			norms = tf.norm(self.features, axis=1)  # (128, 10, 16) -> (128, 10)  labels: (128)

			for i in range(self.num_classes):
				presence = tf.cast(tf.equal(self.class_labels, i), tf.float32)  # binary vector indicating if this output position should contain the correct label
				loss = presence * tf.square(tf.maximum(0., 0.9 - norms[:, i]) + 0.5 * (1. - presence) * tf.square(tf.maximum(0., norms[:, i] - 0.1)))  # (128)
				avg_loss = tf.reduce_mean(loss)
				self.classifier_loss += avg_loss

			predictions = tf.argmax(norms, axis=1, output_type=tf.int32)
			correct = tf.cast(tf.equal(predictions, self.class_labels), tf.float32)
			self.classifier_accuracy = tf.reduce_mean(correct, name='classifier_mean_of_correct')

	def build_autoencoder_loss(self):
		with tf.variable_scope("autoencoder_loss") as scope:
			if self.ae_cost_type == 'mse':
				print "\nUsing MSE loss for decoder"
				self.ae_loss = tf.reduce_mean(tf.square(self.orig_input - self.reconstructed))

			elif self.ae_cost_type == 'ce':
				print "\n\nUsing Cross-Entropy loss\n\n"
				self.ae_loss = -tf.reduce_mean(self.orig_input * tf.log(self.reconstructed + 0.0000001) + \
											   (1 - self.orig_input) * tf.log(1 - self.reconstructed + 0.0000001))

			elif self.ae_cost_type == 'nll':
				print "\n\nUsing NLL loss\n\n"
				self.ae_loss = -tf.reduce_mean(self.orig_input * tf.log(self.reconstructed + 0.0000001))

	def build_model_loss(self):
		self.build_classifier_loss()
		self.build_autoencoder_loss()
		self.model_loss = self.classifier_loss + self.ae_weight * self.ae_loss

	def build_model_training(self):
		with tf.variable_scope("model_training"):
			optimizer = tf.train.AdamOptimizer(self.LR, beta1=self.beta1)
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.model_train_op = optimizer.minimize(self.model_loss)

			if self.debug:
				print "\n\nModel variables and gradients:\n"
				self.model_grads = print_grads_and_vars(optimizer, self.model_loss, tf.trainable_variables(), print_names=True)


	def check_test_loss(self, sess, num_batches):
		batch_losses = []
		batch_number = 0
		test_cifar = inf_test_gen(self)

		while batch_number < num_batches:
			test_images, test_labels = test_cifar.next()
			orig_test_images = np.copy(test_images)

			total_loss, cl_loss, ae_loss = sess.run([self.model_loss, self.classifier_loss, self.ae_loss],
				feed_dict={self.encoder_input: test_images, self.orig_input: orig_test_images, self.class_labels: test_labels})

			batch_losses.append([total_loss, cl_loss, ae_loss])
			batch_number += 1

		return np.mean(batch_losses, axis=0)

	def check_test_accuracy(self, sess):
		batch_number = 0
		batch_accuracies = []

		test_cifar = inf_test_gen(self)

		while batch_number < self.num_test_batches:
			test_input, test_labels = test_cifar.next()

			accuracy = sess.run(self.classifier_accuracy, feed_dict={self.encoder_input: test_input,
																	 self.class_labels: test_labels})

			batch_accuracies.append(accuracy)
			batch_number += 1

		return np.mean(batch_accuracies)


	def train(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True

		with tf.Session(config=config, graph=self.graph) as sess:

			sess.run(self.model_init_op)

			train_cifar = inf_train_gen(self)

			print "\n\nTraining model for {:d} epochs, LR = {:.4f}, {} training batches, {} test batches\n".format(
				self.epochs, self.LR, self.num_train_batches, self.num_test_batches)

			epoch = 0

			while epoch < self.epochs:

				train_batch_number = 0
				train_batch_losses = []
				train_batch_accuracies = []

				while train_batch_number < self.num_train_batches:
					train_images, train_labels = train_cifar.next()
					orig_train_images = np.copy(train_images)

					_, train_loss, train_accuracy = sess.run([self.model_train_op, self.model_loss, self.classifier_accuracy],
										feed_dict={self.encoder_input: train_images,
								   		self.orig_input: orig_train_images, self.class_labels: train_labels})

					train_batch_losses.append(train_loss)  # will be list of zeros for classifier unless debugging
					train_batch_accuracies.append(train_accuracy)  # will be list of zeros for autoencoder
					train_batch_number += 1

				train_batch_losses.append(train_loss)
				train_batch_accuracies.append(train_accuracy)

				avg_train_loss = np.mean(train_batch_losses)
				avg_train_accuracy = np.mean(train_batch_accuracies)
				avg_test_accuracy = self.check_test_accuracy(sess)
				total_test_loss, cl_loss, avg_test_ae_loss = self.check_test_loss(sess, self.num_test_batches)

				print(
				"Epoch {:d}/{:d}: {} | Loss: train {:.5f}, test {:.5f} (cl {:.3f} ae {:.3f}) | Accuracy: train {:.3f} test {:.3f}".format(
					epoch + 1, self.epochs, str(datetime.now())[11:-7], avg_train_loss, total_test_loss, cl_loss,
					self.ae_weight * avg_test_ae_loss, avg_train_accuracy, avg_test_accuracy))

				epoch += 1

		sess.close()

CapsulesNet = Model()
CapsulesNet.train()
