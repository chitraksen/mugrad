package main

import (
	"fmt"
	"math"
	"strings"
)

type Value struct {
	Label    string
	Data     float64
	Grad     float64
	Children []*Value
	Op       string
	Backward func(*Value)
}

func Val(data float64, label string) *Value {
	// create default Value variable
	createValue := Value{label, data, 0, make([]*Value, 0), " ", nil}
	return &createValue
}

func (v *Value) String() string {
	// stringify Value Data
	var info strings.Builder
	info.WriteString(fmt.Sprintf("%v - Data=%v, Grad=%v", v.Label, v.Data, v.Grad))
	if len(v.Children) > 0 {
		info.WriteString(fmt.Sprintf(", Children=%v", v.Children))
	}
	if v.Op != " " {
		info.WriteString(fmt.Sprintf(", Op=%v", string(v.Op)))
	}
	return fmt.Sprintf("Value(%v)", info.String())
}

func Add(a *Value, b *Value, label string) *Value {
	res := Value{label, a.Data + b.Data, 0, []*Value{a, b}, "+", nil}
	// assume two arguments for now
	res.Backward = func(val *Value) {
		val.Children[0].Grad += val.Grad
		val.Children[1].Grad += val.Grad
	}
	return &res
}

func AddConst(a *Value, num float64, label string) *Value {
	b := Val(num, "const")
	return Add(a, b, label)
}

func Sub(a *Value, b *Value, label string) *Value {
	res := Value{label, a.Data - b.Data, 0, []*Value{a, b}, "-", nil}
	res.Backward = func(val *Value) {
		val.Children[0].Grad += val.Grad
		val.Children[1].Grad += val.Grad
	}
	return &res
}

func SubConst(a *Value, num float64, label string) *Value {
	b := Val(num, "const")
	return Sub(a, b, label)
}

func Mul(a *Value, b *Value, label string) *Value {
	res := Value{label, a.Data * b.Data, 0, []*Value{a, b}, "*", nil}
	// assume two arguments for now
	res.Backward = func(val *Value) {
		val.Children[0].Grad += val.Children[1].Data * val.Grad
		val.Children[1].Grad += val.Children[0].Data * val.Grad
	}
	return &res
}

func MulConst(a *Value, num float64, label string) *Value {
	b := Val(num, "const")
	return Mul(a, b, label)
}

func Div(a *Value, b *Value, label string) *Value {
	b_1 := Pow(b, -1, "")
	return Mul(a, b_1, label)
}

func DivConst(a *Value, num float64, label string) *Value {
	b := Val(num, "const")
	return Div(a, b, label)
}

func Pow(x *Value, k float64, label string) *Value {
	res := Value{label, math.Pow(x.Data, k), 0, []*Value{x}, fmt.Sprintf("^ %v", k), nil}
	res.Backward = func(val *Value) {
		derivative := k * (math.Pow(val.Children[0].Data, k-1))
		val.Children[0].Grad += derivative * val.Grad
	}
	return &res
}

func Exp(x *Value, label string) *Value {
	res := Value{label, math.Exp(x.Data), 0, []*Value{x}, "exp", nil}
	res.Backward = func(val *Value) {
		val.Children[0].Grad += val.Data * val.Grad
	}
	return &res
}

func Tanh(x *Value, label string) *Value {
	two_x := 2 * x.Data
	data := (math.Pow(math.E, two_x) - 1) / (math.Pow(math.E, two_x) + 1)
	res := Value{label, data, 0, []*Value{x}, "Tanh", nil}
	// res := math.Tanh(x.Data)
	// assume two arguments for now
	res.Backward = func(val *Value) {
		derivative := 1 - math.Pow(math.Tanh(val.Children[0].Data), 2)
		val.Children[0].Grad += derivative * val.Grad
	}
	return &res
}

func (x *Value) Backprop() {
	var topology []*Value
	visited := map[*Value]bool{}

	var buildTopology func(val *Value)
	buildTopology = func(val *Value) {
		// only add to topology if doesn't already exist
		// don't want multiple passes due to additive nature of grad calc
		if !visited[val] {
			visited[val] = true
			for _, child := range val.Children {
				buildTopology(child)
			}
			topology = append(topology, val)
		}
	}

	// build topology and initialise head node
	buildTopology(x)
	x.Grad = 1

	// reverse topology order backpropagation
	for i := len(topology) - 1; i >= 0; i-- {
		node := topology[i]
		// if node has children, backpropagate gradients to children
		// Backward will only be nil when no children
		if node.Backward != nil {
			node.Backward(node)
		}
	}
}

// EXAMPLE USE CASE - UNCOMMENT TO RUN
// func example1() {
// 	x1 := Val(2, "x1")
// 	x2 := Val(0, "x2")
// 	w1 := Val(-3, "w1")
// 	w2 := Val(1, "w2")
// 	b := Val(6.8813735970195432, "b")
// 	x1w1 := Mul(x1, w1, "x1*w1")
// 	x2w2 := Mul(x2, w2, "x2*w2")
// 	x1w1x2w2 := Add(x1w1, x2w2, "x1*w1 + x2*w2")
// 	n := Add(x1w1x2w2, b, "n")
// 	e := Exp(MulConst(n, 2, "2n"), "e^2n")
// 	o := Div(SubConst(e, 1, "e^2n-1"), AddConst(e, 1, "e^2n+1"), "tanh_sub")
// 	o.Backprop()
// 	VisualiseValue(o)
// }
// func main() {
// 	example1()
// }
