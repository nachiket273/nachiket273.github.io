---
layout: post
title:  "Computational Graph: An Introduction"
date:   2018-08-08 01:00:12 +0530
categories: jekyll posts
---

# Computational Graphs

Computational Graph is interesting concept as form of it, is used in Neural Networks. Computational Graph consists of connected nodes where node can be:

   1) __Operation__: Oparation feeds output to other operations.<br>
   2) __Variable__: Variable feeds input value to operations.<br>
   3) __Placeholder__: Placeholder is special variable where value is inserted during run.<br>


Output of the operation is called "tensor". Tensor can be array(single or multi-dimentional), matrices or higher-dimentional tensors themselves.

Here is example of computational graph from [this amazing blog](http://colah.github.io/)

![Computational Graph Example]({{site.url}}/assets/tree-def.png){:class="img-responsive"}


```python
%reload_ext autoreload
%autoreload 2
```


```python
import numpy as np
```

Let's Define Class for Graph with operations, variables and placeholders.


```python
class Graph():
    def __init__(self):
        self.operations = []
        self.variables = []
        self.placeholders = []

    def defualt(self):
        global _default_graph
        _default_graph = self
```

# Placeholder

These are used to provide runtime input values.<br>
The class also contains the consumers for the placeholder.<br>


```python
class Placeholder():
    def __init__(self):
        self.consumers = []
        _default_graph.placeholders.append(self)
```

# Variables

These are the inputs used by operations<br>
The class also contains the consumers for variables <br>


```python
class Variable():
    def __init__(self, initial_value=None):
        self.value = initial_value
        self.consumers = []
        _default_graph.variables.append(self)
```

# Operation

Operation has following elements:

   1. __Compute__ : The operation
   2. __Input Nodes__: The variables of other operations to be used as input for compute.
   3. __Consumers__: The nodes which will consume the output of the operation


```python
class Operation():
    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes
        self.consumers = []
        for input_node in input_nodes:
            input_node.consumers.append(self)
    
        _default_graph.operations.append(self)

    def compute(self):
        pass
```

# Elementary Operations - Addition and Matrix Multiplication


```python
class Add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x1 , y1):
        self.inputs=[x1, y1]
        return x1 + y1
```


```python
class MatMul(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x1, y1):
        self.inputs = [x1, y1]
        return x1.dot(y1)
```

# Session Run To Compute Output

We'll now create session class which will encapsulate the execution of compute graph<br>
Run function of Session class will be called to compute<br>
The input to the run function will be operation to be performed ( e in the above graph) and dictionary containing values for placeholders <br>
<br>
<br>
<br>
The operation needs to be calculated in correct order.<br>
We'll use postorder traversal to get correct sequence of operations.<br>
We need to make sure the value of each input for operation O needs to be computed before computing operation O<br>


```python
class Session:
    def run(self, operation, feed_dict = {}):
        nodes_po = traverse_po(operation)
        for node in nodes_po:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else: 
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
                
            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output


def traverse_po(operation):
    nodes_po = []
    recurse(operation, nodes_po)
    return nodes_po

def recurse(node, nodes_po):
    if isinstance(node, Operation):
        for input_node in node.input_nodes:
            recurse(input_node, nodes_po)
    nodes_po.append(node)
```

# Examples

## Single Operation - Addition / Matrix Multiplication


```python
Graph().defualt()

x = Variable([[4,5],[9,5]])
y = Variable([[7,3],[4,8]])
z = Add(x, y)
sess = Session()
res = sess.run(z)
print(res)
```

    [[11  8]
     [13 13]]



```python
Graph().defualt()

x = Variable([[4,5],[9,5]])
y = Variable([[7,3],[4,8]])
z = MatMul(x, y)
sess = Session()
res = sess.run(z)
print(res)
```

    [[48 52]
     [83 67]]


# Use Placeholder


```python
Graph().defualt()

x = Variable([[4,3],[2,1]])
y =  Placeholder()
z = Add(x, y)
sess = Session()
res = sess.run(z, {y:[[1,2],[4,6]]})
print(res)
```

    [[5 5]
     [6 7]]


# Multiple Operations


```python
Graph().defualt()

A = Variable([[1,0],[0,1]])
B = Variable([1,1])
X = Placeholder()
Y = MatMul(X, A)
Z = Add(Y, B)
sess = Session()
res =sess.run(Z, {X: [1, 2]})
print(res)
```

    [2 3]


# References:
1. [http://colah.github.io/](http://colah.github.io/)
2. [http://www.deepideas.net](http://www.deepideas.net)


# Fin !!!
