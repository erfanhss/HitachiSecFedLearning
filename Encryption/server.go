// Package main is the entry-point for the go-sockets server sub-project.
// The go-sockets project is available under the GPL-3.0 License in LICENSE.
package main

import (
	"C"
	"bufio"
	"fmt"
	"time"

	//"fmt"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/dckks"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/utils"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	//"time"
)

var numPeers int

var cpkgWrite []sync.Mutex
var pkStrWrite []sync.Mutex
var done []sync.Mutex

var encryptionWrite []sync.Mutex
var liveStatusWrite []sync.Mutex
var pcksShareWrite []sync.Mutex
var encryptionResWrite []sync.Mutex
var decryptionParticipationWrite []sync.Mutex
var tpkWrite []sync.Mutex

var publicKeyStr string
var cpkgSharesStr []string

var targetPublicKeyStr string
var encResultStr string

var encClientInputs []string
var pcksSharesStr []string
var decryptionCoefficients []uint64
var id []uint64

var decryptionParticipation []int
var liveStatus []int

type SafeStringArray struct {
	mu          sync.Mutex
	stringArray []string
	numContents int
}

func (c *SafeStringArray) Update(str string) {
	c.mu.Lock()
	c.stringArray[c.numContents] = str
	c.numContents++
	c.mu.Unlock()
}

//export serverPhase1
func serverPhase1(serverAddress string, numPeers int, robust bool, logDegree uint64, scale float64) {
	// Start the server and listen for incoming connections.

	var shareCounters []SafeStringArray
	var ringPrime uint64 = 0x10000000001d0001
	var ringPrimeP uint64 = 0xfffffffffffc001
	//startTime := time.Now()
	//fmt.Println("Starting " + "tcp" + " server on " + serverAddress)
	l, err := net.Listen("tcp", serverAddress)
	if err != nil {
		fmt.Println("Error listening:", err.Error())
		os.Exit(1)
	}
	// Close the listener when the application closes.
	defer l.Close()
	// The array to save the address of clients
	clientIPs := make([]net.Conn, numPeers)
	for cntr := 0; cntr < numPeers; cntr++ {
		// Listen for an incoming connection.
		c, err := l.Accept()
		if err != nil {
			fmt.Println("Error connecting:", err.Error())
			return
		}
		// Print client connection address.
		//fmt.Println("Client " + c.RemoteAddr().String() + " connected.")
		// Add connection to list
		clientIPs[cntr] = c
	}

	////// Initialize HE
	moduli := &ckks.Moduli{Qi: []uint64{ringPrime}, Pi: []uint64{ringPrimeP}}
	params, err := ckks.NewParametersFromModuli(logDegree, moduli)
	shareCounters = make([]SafeStringArray, numPeers)
	for counter := range shareCounters {
		shareCounters[counter].stringArray = make([]string, numPeers)
		shareCounters[counter].numContents = 0
	}

	params.SetScale(scale)
	params.SetLogSlots(logDegree - 1)
	lattigoPRNG, err := utils.NewKeyedPRNG([]byte{'l', 'a', 't', 't', 'i', 'g', 'o'})
	if err != nil {
		panic(err)
	}
	// Ring for the common reference polynomials sampling
	ringQP, _ := ring.NewRing(params.N(), append(params.Qi(), params.Pi()...))
	//Common reference polynomial generator that uses the PRNG
	crsGen := ring.NewUniformSampler(lattigoPRNG, ringQP)
	crs := crsGen.ReadNew() // for the public-key
	// Target private and public keys
	ckg := dckks.NewCKGProtocol(params)

	cpkgShares := make([]dckks.CKGShare, numPeers)
	cpkgSharesStr = make([]string, numPeers)
	cpkgWrite = make([]sync.Mutex, numPeers)
	done = make([]sync.Mutex, numPeers)
	pkStrWrite = make([]sync.Mutex, numPeers)
	////// Initialize handling clients
	for peerIdx := range cpkgWrite {
		pkStrWrite[peerIdx].Lock()
		cpkgWrite[peerIdx].Lock()
		done[peerIdx].Lock()

	}
	for idx := 0; idx < numPeers; idx++ {
		go handleClientPhase1(clientIPs, idx, robust, &shareCounters)
	}

	////// Wait until key generation shares are uploaded and then generate the collective key
	for peerIdx := range cpkgWrite {
		cpkgWrite[peerIdx].Lock()
	}

	pk := ckks.NewPublicKey(params)
	ckgCombined := ckg.AllocateShares()
	for peerIdx := range clientIPs {
		coeffs := polyCoeffsDecode(cpkgSharesStr[peerIdx])
		poly := ringQP.NewPoly()
		poly.SetCoefficients(coeffs)

		cpkgShares[peerIdx] = poly

		ckg.AggregateShares(cpkgShares[peerIdx], ckgCombined, ckgCombined)

	}
	ckg.GenPublicKey(ckgCombined, crs, pk)

	publicKeyStr = ""
	pkContent := pk.Get()
	for itemIdx := range pkContent {
		publicKeyStr += polyCoeffsEncode(pkContent[itemIdx].Coeffs) + "/"
	}
	publicKeyStr += "\n"
	for peerIdx := range cpkgWrite {
		pkStrWrite[peerIdx].Unlock()
	}

	for peerIdx := range done {
		done[peerIdx].Lock()
		done[peerIdx].Unlock()
	}
}
func handleClientPhase1(connections []net.Conn, idx int, robust bool, shareCounters *[]SafeStringArray) {

	////// transmit the index of the client and number of peers

	numPeers := len(connections)
	conn := connections[idx]
	conn.Write([]byte(strconv.Itoa(idx) + " " + strconv.Itoa(numPeers) + "\n"))
	if robust {
		////// receive shamir shares and relay
		// collect shares
		message, err := bufio.NewReader(conn).ReadString('\n')
		if err != nil {
			panic(err)
		}

		sharesArr := strings.Split(message, ":")
		for partyCntr := 0; partyCntr < numPeers; partyCntr++ {
			sharePacket := sharesArr[partyCntr]
			sharedParts := strings.Split(sharePacket, "/")
			////fmt.Println(len(sharedParts))
			idxToGo, _ := strconv.Atoi(sharedParts[0])
			share := sharedParts[1]
			(*shareCounters)[idxToGo].Update(share)
		}

		for {
			time.Sleep(10 * time.Microsecond)
			if (*shareCounters)[0].numContents == numPeers {
				break
			}
		}
		// relay shares
		toGoStr := strings.Join((*shareCounters)[idx].stringArray, ":") + "\n"
		conn.Write([]byte(toGoStr))

	}

	/////// Receive shares and generate collective public key
	message, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		panic(err)
	}
	cpkgSharesStr[idx] = message[0 : len(message)-1]
	cpkgWrite[idx].Unlock()
	pkStrWrite[idx].Lock()
	conn.Write([]byte(publicKeyStr))
	done[idx].Unlock()
}

//export serverPhase2
func serverPhase2(serverAddress string, numPeers int, robust bool, resiliency float64, logDegree uint64, scale float64, inputLength int) (result *C.char) {
	// Start the server and listen for incoming connections.
	var ringPrime uint64 = 0x10000000001d0001
	var ringPrimeP uint64 = 0xfffffffffffc001
	//fmt.Println("Starting " + "tcp" + " server on " + serverAddress)
	l, err := net.Listen("tcp", serverAddress)
	if err != nil {
		//fmt.Println("Error listening:", err.Error())
		os.Exit(1)
	}
	// Close the listener when the application closes.
	defer l.Close()
	// The array to save the address of clients
	clientIPs := make([]net.Conn, numPeers)
	// run loop forever, until exit.
	for cntr := 0; cntr < numPeers; cntr++ {
		// Listen for an incoming connection.
		c, err := l.Accept()
		if err != nil {
			//fmt.Println("Error connecting:", err.Error())
			return
		}
		// Print client connection address.
		//fmt.Println("Client " + c.RemoteAddr().String() + " connected.")
		// Add connection to list
		clientIPs[cntr] = c
	}
	threshold := int(float64(numPeers) * (1 - resiliency))
	if robust == false {
		threshold = numPeers
	}

	////// Initialize HE
	moduli := &ckks.Moduli{Qi: []uint64{ringPrime}, Pi: []uint64{ringPrimeP}}
	params, err := ckks.NewParametersFromModuli(logDegree, moduli)

	params.SetScale(scale)
	params.SetLogSlots(logDegree - 1)

	numPieces := 0
	packSize := 2 * int(params.Slots())
	if inputLength%packSize == 0 {
		numPieces = inputLength / packSize
	} else {
		numPieces = inputLength/packSize + 1
	}
	if err != nil {
		panic(err)
	}
	// Ring for the common reference polynomials sampling
	ringQ, _ := ring.NewRing(params.N(), params.Qi())
	// Target private and public keys
	tsk, tpk := ckks.NewKeyGenerator(params).GenKeyPair()
	pcks := dckks.NewPCKSProtocol(params, 3.19)

	pcksSharesStr = make([]string, numPeers)
	encClientInputs = make([]string, numPeers)
	pcksShares := make([][]dckks.PCKSShare, numPeers)
	encInputs := make([][]*ckks.Ciphertext, numPeers)
	liveStatus = make([]int, numPeers)
	id = make([]uint64, numPeers)

	encryptionWrite = make([]sync.Mutex, numPeers)
	liveStatusWrite = make([]sync.Mutex, numPeers)
	pcksShareWrite = make([]sync.Mutex, numPeers)
	encryptionResWrite = make([]sync.Mutex, numPeers)
	decryptionParticipationWrite = make([]sync.Mutex, numPeers)
	tpkWrite = make([]sync.Mutex, numPeers)
	////// Initialize handling clients
	//fmt.Println("Transmitting network details to clients")
	for i := 0; i < numPeers; i++ {
		encryptionResWrite[i].Lock()
		decryptionParticipationWrite[i].Lock()
		tpkWrite[i].Lock()
		encryptionWrite[i].Lock()
		done[i].Lock()
		liveStatusWrite[i].Lock()
		pcksShareWrite[i].Lock()

	}
	for idx := 0; idx < numPeers; idx++ {
		go handleClientPhase2(clientIPs, idx)
	}

	////// Wait until encryption results are filled in
	for i := 0; i < numPeers; i++ {
		encryptionWrite[i].Lock()
	}
	evaluator := ckks.NewEvaluator(params)
	for encCounter := range encClientInputs {
		encInputs[encCounter] = make([]*ckks.Ciphertext, numPieces)
		crtClient := encClientInputs[encCounter]
		piecesArr := strings.Split(crtClient, ":")
		piecesArr = piecesArr[0 : len(piecesArr)-1]
		if len(piecesArr) != numPieces {
			fmt.Println("Encryted files received incorrectly")
		}
		for pieceCounter := range encInputs[encCounter] {
			crt := piecesArr[pieceCounter]
			polyCoeffsStringArr := strings.Split(crt, "/")
			polyCoeffsStringArr = polyCoeffsStringArr[0 : len(polyCoeffsStringArr)-1]
			ctContents := make([]*ring.Poly, len(polyCoeffsStringArr))
			for ctContentCounter := range ctContents {
				ctContents[ctContentCounter] = ring.NewPoly(params.N(), params.MaxLevel()+1)
				ctContents[ctContentCounter].SetCoefficients(polyCoeffsDecode(polyCoeffsStringArr[ctContentCounter]))
			}
			encInputs[encCounter][pieceCounter] = ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
			encInputs[encCounter][pieceCounter].SetValue(ctContents)
		}
	}
	encResult := make([]*ckks.Ciphertext, numPieces)
	for pieceCounter := range encResult {
		encResult[pieceCounter] = ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
		evaluator.Add(encInputs[0][pieceCounter], encInputs[1][pieceCounter], encResult[pieceCounter])
		for i := 2; i < numPeers; i++ {
			evaluator.Add(encResult[pieceCounter], encInputs[i][pieceCounter], encResult[pieceCounter])
		}
	}
	encResultStr = ""
	for pieceCounter := range encResult {
		for ctPolyCounter := range encResult[pieceCounter].Value() {
			encResultStr += polyCoeffsEncode(encResult[pieceCounter].Value()[ctPolyCounter].Coeffs) + "/"
		}
		encResultStr += ":"
	}
	encResultStr += "\n"
	//fmt.Println("Encrypted inputs aggregated")

	for idx := 0; idx < numPeers; idx++ {
		encryptionResWrite[idx].Unlock()
	}

	////// Wait until liveness results come in

	for idx := 0; idx < numPeers; idx++ {
		liveStatusWrite[idx].Lock()
	}
	//fmt.Println("Liveness status data received")
	// select a subset of active users
	decryptionParticipation = make([]int, numPeers)
	evalPointsParticipated := make([]uint64, threshold)
	decryptionCoefficients = make([]uint64, numPeers)

	if robust {
		tmpCntr := 0
		for peerIdx := range clientIPs {
			if liveStatus[peerIdx] == 1 {
				if tmpCntr < threshold {
					decryptionParticipation[peerIdx] = 1
					evalPointsParticipated[tmpCntr] = id[peerIdx]
					tmpCntr++
				} else {
					break
				}
			}
		}
		//fmt.Println("Active users selected")
		tmpCntr = 0
		tmpCoefficients := GenerateVandermondeInverse(evalPointsParticipated, ringPrime)
		for peerIdx := range clientIPs {
			if liveStatus[peerIdx] == 1 {
				if tmpCntr < threshold {
					decryptionCoefficients[peerIdx] = tmpCoefficients[tmpCntr]
					tmpCntr++
				} else {
					break
				}
			}
		}
		//fmt.Println("Inverse Vandermonde coefficients generated")
		//fmt.Println(decryptionCoefficients)

	} else {
		for peerIdx := range clientIPs {
			decryptionParticipation[peerIdx] = 1
			decryptionCoefficients[peerIdx] = 0
		}
	}
	for idx := 0; idx < numPeers; idx++ {
		decryptionParticipationWrite[idx].Unlock()
	}
	targetPublicKeyStr = ""
	targetPKContent := tpk.Get()

	for itemIdx := range targetPKContent {
		targetPublicKeyStr += polyCoeffsEncode(targetPKContent[itemIdx].Coeffs) + "/"
	}
	targetPublicKeyStr += "\n"
	for idx := 0; idx < numPeers; idx++ {
		tpkWrite[idx].Unlock()
	}

	////// Wait to collect all pcks shares
	//fmt.Println("Collective key switch share collected")
	pcksCombined := make([]dckks.PCKSShare, numPieces)
	for i := range pcksCombined {
		pcksCombined[i] = pcks.AllocateShares(params.MaxLevel())
	}
	for idx := 0; idx < numPeers; idx++ {
		if decryptionParticipation[idx] == 1 {
			pcksShareWrite[idx].Lock()
		}
	}

	for peerIdx := range pcksShares {
		if decryptionParticipation[peerIdx] == 1 {
			pcksShares[peerIdx] = make([]dckks.PCKSShare, numPieces)
			crtStrPiece := pcksSharesStr[peerIdx]
			crtStrPieceArr := strings.Split(crtStrPiece, ":")
			crtStrPieceArr = crtStrPieceArr[0 : len(crtStrPieceArr)-1]
			if len(crtStrPieceArr) != numPieces {
				fmt.Println("pcks error!")
			}
			for pieceCounter := range pcksShares[peerIdx] {
				crtStr := crtStrPieceArr[pieceCounter]
				polyCoeffStr := strings.Split(crtStr, "/")
				pcksShares[peerIdx][pieceCounter] = pcks.AllocateShares(params.MaxLevel())
				for contentCounter := range pcksShares[peerIdx][pieceCounter] {
					pcksShares[peerIdx][pieceCounter][contentCounter] = ringQ.NewPolyLvl(params.MaxLevel())
					pcksShares[peerIdx][pieceCounter][contentCounter].SetCoefficients(polyCoeffsDecode(polyCoeffStr[contentCounter]))
				}
				pcks.AggregateShares(pcksShares[peerIdx][pieceCounter], pcksCombined[pieceCounter], pcksCombined[pieceCounter])
			}
		}
	}
	encOut := make([]*ckks.Ciphertext, numPieces)
	decryptor := ckks.NewDecryptor(params, tsk)
	encoder := ckks.NewEncoder(params)
	ptres := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	output := make([]float64, numPieces*packSize)
	for pieceCounter := range encOut {
		encOut[pieceCounter] = ckks.NewCiphertext(params, 1, params.MaxLevel(), params.Scale())
		pcks.KeySwitch(pcksCombined[pieceCounter], encResult[pieceCounter], encOut[pieceCounter])
		decryptor.Decrypt(encOut[pieceCounter], ptres)
		tmp := encoder.Decode(ptres, params.Slots())
		for i := 0; i < int(params.Slots()); i++ {
			output[pieceCounter*packSize+i*2] = real(tmp[i])
			output[pieceCounter*packSize+i*2+1] = imag(tmp[i])
		}
	}
	output = output[0:inputLength]
	//fmt.Println(output[19990:])
	outputStr := ""
	for idx := range output {
		outputStr += fmt.Sprintf("%f", output[idx]) + " "
	}

	result = C.CString(outputStr)
	//fmt.Println("Time spent: ", time.Since(startTime))
	for i := range done {
		done[i].Lock()

	}
	for i := 0; i < numPeers; i++ {
		encryptionResWrite[i].Unlock()
		decryptionParticipationWrite[i].Unlock()
		encryptionWrite[i].Unlock()
		done[i].Unlock()
		liveStatusWrite[i].Unlock()
		if decryptionParticipation[i] == 1 {
			tpkWrite[i].Unlock()
			pcksShareWrite[i].Unlock()
		}
	}
	return
}

func handleClientPhase2(connections []net.Conn, idx int) {
	////// transmit the index of the client and number of peers

	conn := connections[idx]
	conn.Write([]byte(strconv.Itoa(idx) + "\n"))
	////// Receive encrypted inputs and aggregate
	message, err := bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		panic(err)
	}
	encClientInputs[idx] = message[0 : len(message)-1]
	encryptionWrite[idx].Unlock()
	encryptionResWrite[idx].Lock()
	conn.Write([]byte(encResultStr))
	////// Receive liveness status and let the chosen nodes know
	message, err = bufio.NewReader(conn).ReadString('\n')
	if err != nil {
		panic(err)
	}
	message = message[0 : len(message)-1]
	messageArr := strings.Split(message, " ")
	liveStatus[idx], _ = strconv.Atoi(messageArr[0])
	id[idx], _ = strconv.ParseUint(messageArr[1], 10, 64)
	liveStatusWrite[idx].Unlock()
	decryptionParticipationWrite[idx].Lock()
	toGoStr := strconv.Itoa(decryptionParticipation[idx]) + " " + strconv.FormatUint(decryptionCoefficients[idx], 10) + " " + "\n"
	conn.Write([]byte(toGoStr))
	if decryptionParticipation[idx] == 1 {
		tpkWrite[idx].Lock()

		//fmt.Println("Target public key transmitted")
		conn.Write([]byte(targetPublicKeyStr))
		////// Receive the decryption shares
		message, err = bufio.NewReader(conn).ReadString('\n')
		if err != nil {
			panic(err)
		}
		pcksSharesStr[idx] = message[0 : len(message)-1]
		pcksShareWrite[idx].Unlock()
	}
	done[idx].Unlock()

}
