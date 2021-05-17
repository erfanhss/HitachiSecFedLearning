import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mpi4py import MPI
import argparse
import time
import model


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
model = model.Model(learning_rate)
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

