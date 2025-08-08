// The minimum runtime for the Klarer language.
package main

import (
	"os"
	"strconv"
)

func Print(s string) {
	os.Stdout.WriteString(s + "\n")
}

func IntToStr(i int) string {
	return strconv.Itoa(i)
}

func BoolToStr(i int) string {
	if i == 1 {
		return "true"
	} else {
		return "false"
	}
}

func CharToStr(c int) string {
	return string(rune(c))
}
