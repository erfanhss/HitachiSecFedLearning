package main

import (
	"C"
	"bufio"
	"fmt"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
	"math/rand"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

type party struct {
	sk           []*ckks.SecretKey
	secretShares []*ring.Poly
	shamirShare  *ring.Poly
	ckgShare     dckks.CKGShare
	pcksShare    []dckks.PCKSShare
	input        [][]complex128
}

var encInput []*ckks.Ciphertext

//export clientPhase1
func clientPhase1(serverAddress string, robust bool, logDegree uint64, scale float64, resiliency float64) (cpkString *C.char, shamirShareString *C.char, id int) {
	//func clientPhase1(serverAddress string, robust bool, logDegree uint64, scale float64, resiliency float64) (cpkString string, shamirShareString string, id int) {

	var ringPrime uint64 = 0x10000000001d0001
	var ringPrimeP uint64 = 0xfffffffffffc001
	///////// Start the client and connect to the server.
	//startTime := time.Now()
	rand.Seed(time.Now().UTC().UnixNano())
	//fmt.Println("Connecting to", "tcp", "server", serverAddress)
	conn, err := net.Dial("tcp", serverAddress)
	if err != nil {
		fmt.Println("Error connecting:", err.Error())
		os.Exit(1)
	}
	defer conn.Close()
	///////// receive network parameters from server
	message, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		panic(err)
	}
	messageUnpacked := strings.Split(message[0:len(message)-1], " ")
	id, _ = strconv.Atoi(messageUnpacked[0])
	numPeers, _ = strconv.Atoi(messageUnpacked[1])
	k := int(float64(numPeers) * (1 - resiliency))
	if k == 0 {
		k = 1
	}

	///////// Initialize HE
	moduli := &ckks.Moduli{Qi: []uint64{ringPrime}, Pi: []uint64{ringPrimeP}}
	params, err := ckks.NewParametersFromModuli(logDegree, moduli)
	params.SetScale(scale)
	params.SetLogSlots(logDegree - 1)
	lattigoPRNG, err := utils.NewKeyedPRNG([]byte{'l', 'a', 't', 't', 'i', 'g', 'o'})
	if err != nil {
		panic(err)
	}
	///////// Ring for the common reference polynomials sampling
	ringQP, _ := ring.NewRing(params.N(), append(params.Qi(), params.Pi()...))
	///////// Common reference polynomial generator that uses the PRNG
	crsGen := ring.NewUniformSampler(lattigoPRNG, ringQP)
	crs := crsGen.ReadNew() // for the public-key

	///////// Target private and public keys
	ckg := dckks.NewCKGProtocol(params)

	///////// create the party object and setup keys
	pi := &party{}
	if robust {
		evalPoints := make([]uint64, numPeers)
		for i := 0; i < numPeers; i++ {
			evalPoints[i] = uint64(i) + 1
		}
		pi.sk = make([]*ckks.SecretKey, k, k)
		pi.secretShares = make([]*ring.Poly, numPeers, numPeers)
		///////// create k different secret keys for each party
		for partyCntr := 0; partyCntr < k; partyCntr++ {
			pi.sk[partyCntr] = ckks.NewKeyGenerator(params).GenSecretKey()
		}
		///////// create the shares of the secret key
		//fmt.Println("Generating shamir shares")
		for partyCntr := 0; partyCntr < numPeers; partyCntr++ {
			vandermonde := GenerateVandermonde(evalPoints[partyCntr], uint64(k), ringPrime)
			res := ringQP.NewPoly()
			ringQP.MulScalar(pi.sk[0].Get(), vandermonde[0], res)
			for i := 1; i < k; i++ {
				tmp := ringQP.NewPoly()
				ringQP.MulScalar(pi.sk[i].Get(), vandermonde[i], tmp)
				ringQP.Add(tmp, res, res)
			}
			pi.secretShares[partyCntr] = res
		}
	} else {
		pi.sk = make([]*ckks.SecretKey, 1, 1)
		///////// create k different secret keys for each party
		pi.sk[0] = ckks.NewKeyGenerator(params).GenSecretKey()
	}
	///////// Create party, and allocate the memory for all the shares that the protocols will need
	pi.ckgShare = ckg.AllocateShares()
	if robust {
		////// transmit shamir shares to others
		toSendString := ""
		for clientIndex, share := range pi.secretShares {
			coeffString := polyCoeffsEncode(share.Coeffs)
			toSendString += strconv.Itoa(clientIndex) + "/" + coeffString + ":"
		}
		toSendString += "\n"
		conn.Write([]byte(toSendString))
		///////// receive shamir shares from others
		message, err = bufio.NewReader(conn).ReadString('\n')
		if err != nil {
			panic(err)
		}
		shares := strings.Split(message, ":")
		secretShares := make([]*ring.Poly, numPeers)
		for partyCounter := 0; partyCounter < numPeers; partyCounter++ {
			shareStr := shares[partyCounter]
			polyCoeff := polyCoeffsDecode(shareStr)
			secretShares[partyCounter] = ringQP.NewPoly()
			secretShares[partyCounter].SetCoefficients(polyCoeff)
		}
		// generate shamir share of collective secret key
		pi.shamirShare = ringQP.NewPoly()
		ringQP.Add(secretShares[0], secretShares[1], pi.shamirShare)
		for i := 2; i < numPeers; i++ {
			ringQP.Add(pi.shamirShare, secretShares[i], pi.shamirShare)
		}
		//fmt.Println("Shamir share generated successfully")
	}
	///////// Collective public key generation
	ckg.GenShare(pi.sk[0].Get(), crs, pi.ckgShare)
	///////// Transmit collective key generation share to the master
	toSendString := polyCoeffsEncode(pi.ckgShare.Coeffs) + "\n"
	conn.Write([]byte(toSendString))
	//fmt.Println("cpk share transmitted to the master")
	///////// receive collective public key from the master
	message, err = bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		panic(err)
	}
	itemsString := strings.Split(message, "/")
	itemsString = itemsString[0 : len(itemsString)-1]
	if len(itemsString) != 2 {
		fmt.Println("Collective Public Key error")
		return
	}
	var itemsPoly [2]*ring.Poly
	for counter := range itemsPoly {
		tmp := polyCoeffsDecode(itemsString[counter])
		itemsPoly[counter] = ringQP.NewPoly()
		itemsPoly[counter].SetCoefficients(tmp)
	}
	pk := ckks.NewPublicKey(params)
	pk.Set(itemsPoly)

	cpkString = C.CString(message[0 : len(message)-1])
	if robust {
		shamirShareString = C.CString(polyCoeffsEncode(pi.shamirShare.Coeffs))
	} else {
		shamirShareString = C.CString(polyCoeffsEncode(pi.sk[0].Get().Coeffs))
	}
	//cpkString = message[0 : len(message)-1]
	//if robust {
	//	shamirShareString = polyCoeffsEncode(pi.shamirShare.Coeffs)
	//} else {
	//	shamirShareString = polyCoeffsEncode(pi.sk[0].Get().Coeffs)
	//}

	return
}

//export clientPhase2
func clientPhase2(inputs []float64, cpkString string, shamirShareString string, id int, serverAddress string, robust bool, logDegree uint64, scale float64, resiliency float64) {
	var ringPrime uint64 = 0x10000000001d0001
	var ringPrimeP uint64 = 0xfffffffffffc001
	///////// Start the client and connect to the server.
	//startTime := time.Now()
	rand.Seed(time.Now().UTC().UnixNano())
	//fmt.Println("Connecting to", "tcp", "server", serverAddress)
	conn, err := net.Dial("tcp", serverAddress)
	if err != nil {
		//fmt.Println("Error connecting:", err.Error())
		os.Exit(1)
	}
	defer conn.Close()
	///////// receive network parameters from server
	message, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		panic(err)
	}
	//messageUnpacked := message[0:len(message)-1]
	//idCrt, _ := strconv.Atoi(messageUnpacked)
	k := int(float64(numPeers) * (1 - resiliency))
	if k == 0 {
		k = 1
	}
	//fmt.Println("id in this round", idCrt, numPeers, k)

	///////// Initialize HE
	//fmt.Println("Initializing HE")

	moduli := &ckks.Moduli{Qi: []uint64{ringPrime}, Pi: []uint64{ringPrimeP}}
	params, err := ckks.NewParametersFromModuli(logDegree, moduli)
	params.SetScale(scale)
	params.SetLogSlots(logDegree - 1)
	if err != nil {
		panic(err)
	}
	///////// Ring for the common reference polynomials sampling
	ringQP, _ := ring.NewRing(params.N(), append(params.Qi(), params.Pi()...))
	///////// Common reference polynomial generator that uses the PRNG
	pcks := dckks.NewPCKSProtocol(params, 3.19)
	///////// generate evaluation points for secret sharing

	///////// create the party object and setup keys
	pi := &party{}
	pi.shamirShare = ringQP.NewPoly()
	pi.shamirShare.SetCoefficients(polyCoeffsDecode(shamirShareString))
	///////// Create party, and allocate the memory for all the shares that the protocols will need
	// creating inputs
	numPieces := 0
	inputLength := len(inputs)
	packSize := 2 * int(params.Slots())
	if inputLength%packSize == 0 {
		numPieces = inputLength / packSize
	} else {
		numPieces = inputLength/packSize + 1
	}
	//fmt.Println("Working with ", numPieces, " pieces")
	pi.input = make([][]complex128, numPieces)
	for i := 0; i < numPieces; i++ {
		pi.input[i] = make([]complex128, params.Slots())
		for j := range pi.input[i] {
			if i*packSize+2*j < inputLength && i*packSize+2*j+1 < inputLength {
				firstElem := inputs[i*packSize+2*j]
				secElem := inputs[i*packSize+2*j+1]
				pi.input[i][j] = complex(firstElem, secElem)
			} else if i*packSize+2*j < inputLength {
				firstElem := inputs[i*packSize+2*j]
				pi.input[i][j] = complex(firstElem, 0)
			} else {
				pi.input[i][j] = complex(0, 0)
			}

		}
	}
	pi.pcksShare = make([]dckks.PCKSShare, numPieces)
	/// set the collective public key
	itemsString := strings.Split(cpkString, "/")
	itemsString = itemsString[0 : len(itemsString)-1]
	if len(itemsString) != 2 {
		//fmt.Println("Collective Public Key error")
		return
	}
	var itemsPoly [2]*ring.Poly
	for counter := range itemsPoly {
		tmp := polyCoeffsDecode(itemsString[counter])
		itemsPoly[counter] = ringQP.NewPoly()
		itemsPoly[counter].SetCoefficients(tmp)
	}
	pk := ckks.NewPublicKey(params)
	pk.Set(itemsPoly)
	//fmt.Println("Public key installed")
	//startTime = time.Now()

	///////// Encrypt and transmit
	encInput = make([]*ckks.Ciphertext, numPieces)
	encryptor := ckks.NewEncryptorFromPk(params, pk)
	encoder := ckks.NewEncoder(params)
	//fmt.Println("Encryption threads launched")
	for pieceCounter := range encInput {
		encInput[pieceCounter] = ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
		pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
		encoder.Encode(pt, pi.input[pieceCounter], params.Slots())
		encryptor.Encrypt(pt, encInput[pieceCounter])
	}
	//fmt.Println("Encryption done")
	// Transmit encrypted input
	toSendString := ""
	for pieceCounter := range encInput {
		for ctPolyCounter := range encInput[pieceCounter].Value() {
			toSendString += polyCoeffsEncode(encInput[pieceCounter].Value()[ctPolyCounter].Coeffs) + "/"
		}
		toSendString += ":"
	}
	toSendString += "\n"
	conn.Write([]byte(toSendString))
	//fmt.Println("Encryption result sent")
	// receive aggregated encrypted result
	message, err = bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		panic(err)
	}
	pieceArr := strings.Split(message, ":")
	pieceArr = pieceArr[0 : len(pieceArr)-1]
	encResult := make([]*ckks.Ciphertext, numPieces)
	for pieceCounter := range pieceArr {
		message = pieceArr[pieceCounter]
		polyCoeffsStringArr := strings.Split(message, "/")
		polyCoeffsStringArr = polyCoeffsStringArr[0 : len(polyCoeffsStringArr)-1]
		ctContents := make([]*ring.Poly, len(polyCoeffsStringArr))
		for ctContentCounter := range ctContents {
			ctContents[ctContentCounter] = ring.NewPoly(params.N(), params.MaxLevel()+1)
			ctContents[ctContentCounter].SetCoefficients(polyCoeffsDecode(polyCoeffsStringArr[ctContentCounter]))
		}
		encResult[pieceCounter] = ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
		encResult[pieceCounter].SetValue(ctContents)

	}

	//fmt.Println("Received aggregated result")

	////// Announce the status and receive aggregated encryption and Vandermonde coefficients
	var state int
	if robust {
		state = 1
	} else {
		state = 1
	}
	stateStr := strconv.Itoa(state)
	idStr := strconv.Itoa(id + 1)
	conn.Write([]byte(stateStr + " " + idStr + "\n"))
	//fmt.Println("Liveness status announced")
	// wait to receive the result
	message, err = bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		panic(err)
	}
	informationStringArr := strings.Split(message, " ")
	participationStatus, _ := strconv.Atoi(informationStringArr[0])
	decryptionCoefficient, _ := strconv.ParseUint(informationStringArr[1], 10, 64)
	if participationStatus == 1 {
		//fmt.Println("Participating in the decryption with coefficient: ", decryptionCoefficient)
		// receive tpk
		message, err = bufio.NewReader(conn).ReadString('\n')
		if err != nil {
			panic(err)
		}
		itemsString := strings.Split(message, "/")
		itemsString = itemsString[0 : len(itemsString)-1]
		if len(itemsString) != 2 {
			//fmt.Println("Target Public Key error")
			return
		}
		var itemsPoly [2]*ring.Poly
		for counter := range itemsPoly {
			tmp := polyCoeffsDecode(itemsString[counter])
			itemsPoly[counter] = ringQP.NewPoly()
			itemsPoly[counter].SetCoefficients(tmp)
		}
		tpk := ckks.NewPublicKey(params)
		tpk.Set(itemsPoly)
		//fmt.Println("Target public key receieved")
		/////// Generate collective public key switch share and transmit
		//fmt.Println("Generating pcks share")
		for pieceCounter := range pi.pcksShare {
			pi.pcksShare[pieceCounter] = pcks.AllocateShares(params.MaxLevel())
			if robust {
				scaledSecretKey := ringQP.NewPoly()
				ringQP.MulScalar(pi.shamirShare, decryptionCoefficient, scaledSecretKey)
				pcks.GenShare(scaledSecretKey, tpk, encResult[pieceCounter], pi.pcksShare[pieceCounter])

			} else {
				pcks.GenShare(pi.shamirShare, tpk, encResult[pieceCounter], pi.pcksShare[pieceCounter])
			}
		}
		toSendString = ""
		for pieceCounter := range pi.pcksShare {
			for i := range pi.pcksShare[pieceCounter] {
				coeffsString := polyCoeffsEncode(pi.pcksShare[pieceCounter][i].Coeffs)
				toSendString += coeffsString + "/"
			}
			toSendString += ":"
		}
		toSendString += "\n"
		conn.Write([]byte(toSendString))
	} else {
		//fmt.Println("Not participating in the decryption")
	}
	//fmt.Println("Time spent for Phase 2: ", time.Since(startTime))

}
