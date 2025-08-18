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

func BoolToStr(b bool) string {
	if b {
		return "true"
	} else {
		return "false"
	}
}

func CharToStr(c int) string {
	return string(rune(c))
}

func Int__to_str(i int) string {
	return strconv.Itoa(i)
}

func Bool__to_str(b bool) string {
	if b {
		return "true"
	} else {
		return "false"
	}
}

func Char__to_str(c int) string {
	return string(rune(c))
}

func Str__to_str(s string) string {
	return s
}
