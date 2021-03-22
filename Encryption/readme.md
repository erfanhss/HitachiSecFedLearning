# Robust Secure Aggregation based on MPHE
This library provides a robust extension over (https://github.com/ldsec/lattigo) distributed CKKS by incorporating Shamir's secret sharing scheme. It contains a server side application and a client side application. 
# Compiling into a Shared Library
The functions are exported into a shared library in order to be used by the Python code. The following code is used to generated the shared library:
```
go build -o func.so -buildmode=c-shared main.go server.go client.go utils.go
```
