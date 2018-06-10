import numpy as np
import cPickle as pickle
import tensorflow as tf


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


def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, padding='SAME',
		   bias=True, name="conv2d", init='he', verbose=False):
	with tf.variable_scope(name):
		if init == "he":
			shape = input_.get_shape().as_list()
			fan_in = shape[-1] * k_h * k_w
			fan_out = output_dim * k_h * k_w / (d_h * d_w)
			stddev = np.sqrt(4. / (fan_in + fan_out))
		elif init == "he_per_CS231n":   #http://cs231n.github.io/neural-networks-2/#init
			#https://github.com/tensorflow/tensorflow/blob/r1.6/tensorflow/python/ops/init_ops.py
			shape = input_.get_shape().as_list()
			fan_in = shape[-1] * k_h * k_w
			stddev = np.sqrt(2. / fan_in)

		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
		if verbose:
			print conv
		if bias:
			biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
		else:
			biases = tf.zeros([output_dim])

		conv = tf.nn.bias_add(conv, biases)

		return conv


def print_grads_and_vars(opt, loss, var_list, print_names=True):
	grads_and_vars = opt.compute_gradients(loss, var_list)
	gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
	if print_names:
		for g in gradients:
			print g
		for v in variables:
			print v
		print "===========\n\n"
	return gradients




def generate_cifar(batch_size, data_dir):

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

		def get_batch():
			# remove randomness below to compare diff runs for params search
			# put it back if averaging multiple runs for each param value

			# np.random.shuffle(batched)
			images, labels = zip(*batched)
			for i in xrange(len(images) / batch_size):
				image_batch = np.copy(images[i * batch_size:(i + 1) * batch_size])
				label_batch = np.copy(labels[i * batch_size:(i + 1) * batch_size])
				yield (image_batch, label_batch)

		return get_batch

	def get_batches(generator):
		while True:
			for images, labels in generator():
				yield images, labels

	train_gen = cifar_generator(['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5'], batch_size, data_dir)
	test_gen = cifar_generator(['test_batch'], batch_size, data_dir)

	train = get_batches(train_gen)
	test = get_batches(test_gen)

	return train, test