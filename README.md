# Secure and Robust Federated Learning based on MPHE

This library implements a novel secure aggregation framework based on multi-party homomorphic encyrption. We assume a setup with central node and N participants. We only require transmission channels between clients and the central node. Initially the clients setup a collective public key. This step is performed only once. Later on, the public key is used to encrypt and transmit gradients. The central node, attempts to decrypt the aggregated gradients by asking clients for additional information. The algorithm is designed to be robust against user dropout. The resiliency of the network is tunable and determined in the first phase.

## Getting Started
The test is designed to run on a cluster. The clients should be able to transmit TCP packets to the central node.

### Prerequisites
The following packages beed to be installed:

[MPI4Py] (https://mpi4py.readthedocs.io/en/stable/)

[Tensorflow] (https://www.tensorflow.org/api_docs/python/tf)


### Running the test

Start cloning the repo by running the following:
```
git clone https://github.com/erfanhss/HitachiSecFedLearning.git
```
Then run the test file by executing the following command:
```
mpirun -n (size of network) -H (server, list of clients) python main.py --num_peers (number of clients) --server_address (IP of the central node) --robust (robustness) --learning_rate (learning rate) --batch_size_per_worker (batch size) --resiliency (resiliency) --num_iterations (number of iterations)
```
num_peers: number of clients in the network

server_address: IP of the central node (example: "localhost:8080")

robust: deteremines the robustness of the network, "True" for robust algorithm.

learning_rate: learning rate of learning algorithm (defaulted to 0.1)

batch_size_per_worker: the batch size for each worker (defaulted to 256)

num_iterations: the number of iterations for the learning algorithm (defaulted to 100)

resiliency: percentage of the network required for the algorithm to operate (defaulted to 1)

### Aggregation functions
The following functions have been implemented in the GoWrappers.py:
```
client_phase1(server_address, robust, log_degree, log_scale, resiliency)
server_phase1(server_address, num_peers, robust, log_degree, log_scale)
client_phase2(inputs, public_key, shamir_share, id, server_address, robust, log_degree, log_scale, resiliency)
server_phase2(server_address, num_peers, robust, resiliency, log_degree, log_scale, input_length)
```
