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

server_address: IP of the central node ("localhost:8080")

robust: deteremines the robustness of the network, "True" for robust algorithm.

learning_rate: learning rate of learning algorithm. Defaulted to 0.1

batch_size_per_worker: the batch size for each worker. Defaulted to 256

num_iterations: the number of iterations for the learning algorithm. Defaulted to 100

resiliency: percentage of the network required for the algorithm to operate. defaulted to 1

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
