// The minimum runtime for the Klarer language.
package main

import (
	"os"
	"strconv"
)

type ToStr interface {
	to_str(obj any) string
}

func Print(obj any) {
	var s string
	switch obj := obj.(type) {
	case string:
		s = obj
	case int:
		s = Int__to_str(obj)
	case bool:
	    s = Bool__to_str(obj)
	case rune:
		s = Char__to_str(obj)
	default:
		panic("cannot print object")
	}
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

func Char__to_str(c rune) string {
	return string(c)
}

func Str__to_str(s string) string {
	return s
}
