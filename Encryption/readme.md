# Robust Secure Aggregation based on MPHE
This library provides a robust extension over (https://github.com/ldsec/lattigo) distributed CKKS by incorporating Shamir's secret sharing scheme. It contains a server side application and a client side application. 
## Compiling into a shared library
The functions are exported into a shared library in order to be used by the Python code. The following code is used to generated the shared library:
```
go build -o func.so -buildmode=c-shared main.go server.go client.go utils.go
```
## List of files
This directory contains the following files:

main.go: creates the main function for testing purposes

client.go: contains the functionalities required in the client side application

server.go: contains the functionalities required in the server side application

utils.go: contains some helper function used through out the code

func_ubuntu.h: the header of the shared library

func_ubuntu.so: the shared library
