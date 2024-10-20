package main

import (
	"fmt"
	"log"
	"os"

	"github.com/goccy/go-graphviz"
	"github.com/goccy/go-graphviz/cgraph"
)

// Recursive function to create nodes and edges
func buildGraph(graph *cgraph.Graph, val *Value, graphviz *graphviz.Graphviz) (*cgraph.Node, error) {
	// Create a unique label for the node with Data and Grad
	label := fmt.Sprintf("%v | data %.4f | grad %.4f", val.Label, val.Data, val.Grad)
	nodeName := fmt.Sprintf("node_%p", val) // unique name based on memory address
	node, err := graph.CreateNode(nodeName)
	if err != nil {
		return nil, err
	}
	node.SetLabel(label)
	node.SetShape("record")

	// If there are children, we need to create an operation node
	if len(val.Children) > 0 {
		// Create a node for the operation
		opNodeName := fmt.Sprintf("op_%p", val)
		opNode, err := graph.CreateNode(opNodeName)
		if err != nil {
			return nil, err
		}
		opNode.SetLabel(val.Op) // Set the operation label (+, *)

		// Connect the operation node to the parent node
		_, err = graph.CreateEdge("to_parent", opNode, node)
		if err != nil {
			return nil, err
		}

		// Connect each child to the operation node and recursively build the graph for the children
		for _, child := range val.Children {
			childNode, err := buildGraph(graph, child, graphviz)
			if err != nil {
				return nil, err
			}
			_, err = graph.CreateEdge(val.Op, childNode, opNode)
			if err != nil {
				return nil, err
			}
		}
	}

	return node, nil
}

func VisualiseValue(root *Value) {
	g := graphviz.New()
	graph, err := g.Graph()
	if err != nil {
		log.Fatal(err)
	}
	graph.SetRankDir("LR")
	graph.SetFontSize(7)
	defer func() {
		graph.Close()
		g.Close()
	}()

	// Build the graph recursively from the final value
	_, err = buildGraph(graph, root, g)
	if err != nil {
		log.Fatal(err)
	}

	// Set up output file
	outputFile := "output.png"
	f, err := os.Create(outputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Render the graph to an image file
	err = g.RenderFilename(graph, graphviz.PNG, outputFile)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Graph has been saved to %s\n", outputFile)
}
