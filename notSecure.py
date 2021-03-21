import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mpi4py import MPI
import argparse
import time

class Model:
    def __init__(self, lr):
        inputs = keras.Input(shape=(28, 28, 1), name="digits")
        conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        pool1 = layers.MaxPool2D()(conv1)
        conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        pool2 = layers.MaxPool2D()(conv2)
        conv3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
        pool3 = layers.MaxPool2D()(conv3)
        flatten = layers.Flatten()(pool3)
        x1 = layers.Dense(64, activation="relu")(flatten)
        outputs = layers.Dense(10, name="predictions")(x1)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.optimizer = keras.optimizers.SGD(learning_rate=lr)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.shapes = []
        self.flat_gradient_shape = []
        self.calculate_gradients(x_train[0:1], y_train[0:1])
        self.accuracy = []
        self.loss = []

    def calculate_gradients(self, x_train, y_train):
        with tf.GradientTape() as tape:
            logits = self.model(x_train, training=True)
            loss_value = self.loss_fn(y_train, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights, )
        result = self.flatten_gradients(grads)
        self.flat_gradient_shape = result.numpy().shape
        return self.flatten_gradients(grads)

    def flatten_gradients(self, gradients):
        flat_grad = []
        shapes = []
        for arr in gradients:
            flat_grad.append(tf.reshape(arr, [-1, 1]))
            shapes.append(tf.shape(arr))
        self.shapes = shapes
        return tf.concat(flat_grad, axis=0)

    def unflatten(self, flat_grad):
        output = []
        cntr = 0
        for shape in self.shapes:
            num_elements = tf.math.reduce_prod(shape)
            params = tf.reshape(flat_grad[cntr:cntr + num_elements, 0], shape)
            params = tf.cast(params, tf.float32)
            cntr += num_elements
            output.append(params)
        return output

    def update_params(self, flat_grad):
        output = self.unflatten(flat_grad)
        self.optimizer.apply_gradients(zip(output, self.model.trainable_weights))
        acc, loss = self.report_performance()
        self.accuracy.append(acc)
        self.loss.append(loss)

    def report_performance(self):
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')
        test_idx = np.random.permutation(len(x_test))
        test_batch_idx = np.array_split(test_idx, 60)
        for batchIdx in test_batch_idx:
            logits = self.model(x_test[batchIdx], training=False)
            lossValue = self.loss_fn(y_test[batchIdx], logits)/len(batchIdx)
            test_accuracy.update_state(y_test[batchIdx], logits)
            test_loss.update_state(lossValue)
        return test_accuracy.result().numpy(), test_loss.result().numpy()

def master():
    ### letting clients know the server is up
    for iteration in range(iterations):
        init = time.time()
        ### update model paramaters for clients
        params = model.flatten_gradients(model.model.get_weights())
        param_req = []
        for worker_idx in range(num_peers):
            param_req.append(comm.Isend(np.ascontiguousarray(params, dtype=float), dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(param_req)
        gradients = [np.zeros(model.flat_gradient_shape) for _ in range(num_peers)]
        grad_req = []
        for worker_idx in range(num_peers):
            grad_req.append(comm.Irecv(gradients[worker_idx], source=worker_idx+1, tag=0))
        MPI.Request.waitall(grad_req)
        model.update_params(np.mean(gradients, axis=0))
        res = model.report_performance()
        print("-----------------------------------")
        print("Iteration: ", iteration + 1)
        print("Accuracy: ", round(res[0] * 100, 3))
        print("Loss: ", round(res[1], 5))
        print("Time spent: ", time.time() - init)
        print("-----------------------------------")


def client():
    ### upadte model parameters
    for iteration in range(iterations):
        weights = np.zeros(model.flat_gradient_shape, float)
        req = comm.Irecv(weights, source=0, tag=0)
        req.Wait()
        weights = model.unflatten(weights)
        model.model.set_weights(weights)
        part = np.random.permutation(len(x_train))[0:batch_size_per_worker]
        grad = model.calculate_gradients(x_train[part], y_train[part]).numpy()
        req = comm.Isend(grad, dest=0, tag=0)
        req.Wait()

parser = argparse.ArgumentParser()
parser.add_argument("--num_peers", required=True)

parser.add_argument("--learning_rate", required=False, default=0.1)
parser.add_argument("--num_iterations", required=False, default=100)
parser.add_argument("--batch_size_per_worker", required=False, default=256)

args = parser.parse_args()
learning_rate = float(args.learning_rate)
iterations = int(args.num_iterations)
batch_size_per_worker = int(args.batch_size_per_worker)
num_peers = int(args.num_peers)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 28, 28, 1)) / 255.
x_test = np.reshape(x_test, (-1, 28, 28, 1)) / 255.
model = Model(learning_rate)
if rank == 0:
    print("----------------------------------------")
    print("Number of clients: ", num_peers)
    print("Learining with rate: ", learning_rate, " for ", iterations, " iterations and batch size of ", batch_size_per_worker)
    print("------------------------------------------")
# print(input_len)
if rank == 0:
    master()
else:
    client()

