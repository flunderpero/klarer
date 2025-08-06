// The minimalistic runtime for the Klarer language.
package main

import "os"

func Print(s string) {
	os.Stdout.WriteString(s + "\n")
}
