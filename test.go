package main

import "fmt"

func main() {
	id := func(a any) any {
		return a
	}
	fmt.Println(id(123))
}
