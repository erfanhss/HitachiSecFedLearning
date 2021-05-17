import numpy as np
import GoWrappers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mpi4py import MPI
import argparse
import time
import model

def random_matrix(p, s, w, seed):
  counts = tf.cast(tf.random.stateless_normal([1], seed, mean= s*w*p*2, stddev=np.sqrt(s*w*2*p*(1-2*p))), dtype=tf.int32)
  rows = tf.random.stateless_uniform(counts, seed, minval=0, maxval=s, dtype=tf.int64)
  cols = tf.random.stateless_uniform(counts, seed, minval=0, maxval=w, dtype=tf.int64)
  vals = tf.cast(tf.random.stateless_binomial(counts, seed, 1, probs=0.5)*2-1, tf.float32)
  return tf.sparse.reorder(tf.SparseTensor(tf.stack([rows, cols], axis=-1), vals, [s, w]))


def master():
    ### initilize Network setup
    print("Initializing phase 1 at master")
    GoWrappers.server_phase1(server_Address, num_peers, robust, log_degree, log_scale)
    ### letting clients know the server is up
    print("Phase 1 completed")
    for iteration in range(iterations):
        init = time.time()
        ### update model paramaters for clients
        params = model.flatten_gradients(model.model.get_weights())
        param_req = []
        for worker_idx in range(num_peers):
            param_req.append(comm.Isend(np.ascontiguousarray(params, dtype=float), dest=worker_idx + 1, tag=0))
        MPI.Request.waitall(param_req)
        update = GoWrappers.server_phase2(server_Address, num_peers, robust, resiliency, log_degree, log_scale,
                                          samples)
        update = np.array(update).reshape(-1, 1)/num_peers
        if compression == 1:
            phi = random_matrix(alpha/2/samples, samples, params_count, seed=tf.constant([iteration, iteration]))
            update = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(phi), tf.convert_to_tensor(update, dtype=tf.float32))
        model.update_params(update)
        res = model.report_performance()
        print("-----------------------------------")
        print("Iteration: ", iteration + 1)
        print("Accuracy: ", round(res[0] * 100, 3))
        print("Loss: ", round(res[1], 5))
        print("Time spent: ", time.time() - init)
        print("-----------------------------------")


def client():
    time.sleep(1)
    ### initilize Network setup
    print("Initializing phase 1 at client id ", rank)
    pk, shamir_share, id = GoWrappers.client_phase1(server_Address, robust, log_degree, log_scale, resiliency)
    print("Phase 1 completed at client id ", rank)
    error = tf.zeros(model.flat_gradient_shape)
    beta = 1/alpha/(r+1+1/alpha)
    ### upadte model parameters
    for iteration in range(iterations):
        weights = np.zeros(model.flat_gradient_shape, float)
        req = comm.Irecv(weights, source=0, tag=0)
        req.Wait()
        weights = model.unflatten(weights)
        model.model.set_weights(weights)
        part = np.random.permutation(len(x_train))[0:batch_size_per_worker]
        grad = model.calculate_gradients(x_train[part], y_train[part])
        error_compensated = learning_rate * grad + error
        if compression == 1:
            phi = random_matrix(alpha / 2 / samples, samples, params_count, seed=tf.constant([iteration, iteration]))
            compressed = beta * tf.sparse.sparse_dense_matmul(phi, error_compensated)
            recov = tf.sparse.sparse_dense_matmul(tf.sparse.transpose(phi), compressed)
            error = error_compensated - recov
        else:
            compressed = error_compensated
        GoWrappers.client_phase2(compressed.numpy().reshape(-1), pk, shamir_share, id, server_Address, robust, log_degree, log_scale, resiliency)


parser = argparse.ArgumentParser()
parser.add_argument("--num_peers", required=True)
parser.add_argument("--server_address", required=True)
parser.add_argument("--robust", required=True)
parser.add_argument("--compression", required=True)

parser.add_argument("--compression_rate", required=False, default=1)
parser.add_argument("--compression_alpha", required=False, default=0.1)
parser.add_argument("--learning_rate", required=False, default=0.1)
parser.add_argument("--num_iterations", required=False, default=100)
parser.add_argument("--batch_size_per_worker", required=False, default=256)
parser.add_argument("--resiliency", required=False, default=0)

args = parser.parse_args()
learning_rate = float(args.learning_rate)
if args.robust == "True":
    robust = True
elif args.robust == "False":
    robust = False
else:
    print("Improper robustness value")
iterations = int(args.num_iterations)
batch_size_per_worker = int(args.batch_size_per_worker)
resiliency = float(args.resiliency)
num_peers = int(args.num_peers)
alpha = float(args.compression_alpha)
compression = int(args.compression)
r = float(args.compression_rate)
if compression == 0:
    r = 1
server_Address = args.server_address.encode()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 28, 28, 1)) / 255.
x_test = np.reshape(x_test, (-1, 28, 28, 1)) / 255.
model = model.Model(learning_rate)
# server_Address = b"localhost:8080"
params_count = 37962
samples = int(params_count/r)
log_degree = 13
log_scale = 40
input_len = model.flat_gradient_shape[0]
if rank == 0:
    print("----------------------------------------")
    print("Master at: ", server_Address)
    print("Number of clients: ", num_peers)
    print("Robustness against dropouts: ", robust, " with resiliency: ", resiliency)
    print("Learining with rate: ", learning_rate, " for ", iterations, " iterations and batch size of ", batch_size_per_worker)
    print("------------------------------------------")
# print(input_len)
if rank == 0:
    master()
else:
    client()

