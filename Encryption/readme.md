# Robust Secure Aggregation based on MPHE
This library provides a robust extension over (https://github.com/ldsec/lattigo) distributed CKKS by incorporating Shamir's secret sharing scheme. It contains a server side application and a client side application. 
```
go build main.go client.go server.go util.go
```
Then the following code can be executed on participants:
```
go run secureaggregation (client/server)
```
