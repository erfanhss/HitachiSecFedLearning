package main

import (
	"C"
	"strconv"
	"strings"
)

func arrayToString(arr []uint64) string {
	init := strconv.FormatUint(arr[0], 10)
	for cntr := 1; cntr < len(arr); cntr++ {
		init = init + "." + strconv.FormatUint(arr[cntr], 10)
	}
	return init
}

func stringToArray(strIn string) []uint64 {
	//fmt.Println("string to arr called")
	//fmt.Println(strIn[0:20])
	elements := strings.Split(strIn, ".")
	//fmt.Println("string to arr allocating")
	resLoc := make([]uint64, len(elements))
	//fmt.Println("string to arr allocated")
	for counter := range resLoc {
		resLoc[counter], _ = strconv.ParseUint(elements[counter], 10, 64)
	}
	//fmt.Println("string to arr returned")
	return resLoc
}

func polyCoeffsEncode(coeffs [][]uint64) string {
	res := ""
	for dimZeroCounter := range coeffs {
		tmp := coeffs[dimZeroCounter]
		tmpRes := arrayToString(tmp)
		res += tmpRes + " "
	}
	return res
}

func polyCoeffsDecode(str string) [][]uint64 {
	//fmt.Println("I am called")
	polyCoeffsArr := strings.Split(str, " ")
	polyCoeffsArr = polyCoeffsArr[0 : len(polyCoeffsArr)-1]
	//fmt.Println("I am allocating")
	res := make([][]uint64, len(polyCoeffsArr))
	//fmt.Println("I allocated")
	for polyCounter := range res {
		tmp := stringToArray(polyCoeffsArr[polyCounter])
		res[polyCounter] = tmp
	}
	//fmt.Println("I am returning")
	return res
}

func Pow(x uint64, y uint64, p uint64) uint64 {
	if y == 0 {
		return 1
	} else {
		var res uint64 = 1
		var i uint64 = 1
		for i = 0; i < y; i++ {
			res = MultiplyNonOverflow(res, x, p)
		}
		return res
	}
}

func MultiplyNonOverflow(x uint64, y uint64, p uint64) uint64 {
	locA := x
	locB := y
	if locB == 1 {
		return locA
	}
	if locA == 1 {
		return locB
	}
	if locB%2 == 0 {
		locA = (2 * locA) % p
		locB = locB / 2
		return MultiplyNonOverflow(locA, locB, p)
	} else {
		return (locA + MultiplyNonOverflow(locA, locB-1, p)) % p
	}

}

func MultiplicativeInverse(x uint64, p uint64) uint64 {
	r_0 := int64(p)
	r_1 := int64(x)
	t_0 := int64(0)
	t_1 := int64(1)
	for {
		q := r_0 / r_1
		new_r := r_0 - q*r_1
		if new_r == 0 {
			break
		}
		new_t := t_0 - q*t_1
		r_0 = r_1
		t_0 = t_1
		r_1 = new_r
		t_1 = new_t
	}
	if t_1 > 0 {
		return uint64(t_1)
	} else {
		return uint64(int64(p) + t_1)
	}
}

func GenerateVandermonde(evalPoint uint64, threshold uint64, p uint64) []uint64 {
	res := make([]uint64, threshold)
	var i uint64 = 0
	for i = 0; i < threshold; i++ {
		res[i] = Pow(evalPoint, i, p)
	}
	return res
}

func SubField(a uint64, b uint64, p uint64) uint64 {
	if a > b {
		return (a - b) % p
	} else {
		return SubField(a+p, b, p)
	}
}

func GenerateVandermondeInverse(evalPoints []uint64, p uint64) []uint64 {
	res := make([]uint64, len(evalPoints))
	var i int
	for i = 0; i < len(evalPoints); i++ {
		var innerRes uint64 = 1
		for m := 0; m < len(evalPoints); m++ {
			if m != i {
				tmp := MultiplyNonOverflow(evalPoints[m], MultiplicativeInverse(SubField(evalPoints[m], evalPoints[i], p), p), p)
				innerRes = MultiplyNonOverflow(tmp, innerRes, p)
			}
		}
		res[i] = innerRes
	}
	return res
}
