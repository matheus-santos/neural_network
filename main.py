# Working with neural networks
# @author: Matheus Cesario <mts.cesario@gmail.com>

# Train data
# Test data
# Parameters
# Output
# Expected output


# Abstracting Input/Output from a cell
class Edge(object):
    """ Edge (input or output of cell) """

    # Properties
    _value = 0
    _weight = 0

    # Ctor
    def __init__(self, value=0):
        self._value = value

    # Setting value
    def set(self, value=0):
        self._value = value

    # Getting value
    def get(self):
        return self._value

    # Getting weight value (theta)
    def getWeight(self):
        return self._weight

    # Setting weight value (theta)
    def setWeight(self, weight=0):
        self._weight = weight


# Abstracting a network's cell
class Cells():
    """ Cell class """

    # Properties
    _inputs = []
    _outputs = []
    _features = []

    _num_inputs = 0  # Total of inputs
    _num_outputs = 0  # Total of outputs
    _label = "default_label"  # Default label

    # Ctor
    def __init__(self, num_inputs=0, num_outputs=0, label="default_cell"):

        self._label = label  # Defining label

        # Creating inputs
        for i in xrange(0, num_inputs):
            self.createInput(0)

        # Creating outputs
        for i in xrange(0, num_outputs):
            self.createOutput(0)

    # Activation function
    def _h(self):

        # Get all inputs
        h = sum([i.get() for i in self._inputs])

        return h

    # Input
    def input(self, index):
        try:
            return self._inputs[index]
        except Exception:
            print("Error: Input {0} does not exists".format(index))
            return None

    # Output
    def output(self, index=0):

        # Refeshing output
        self._outputs[index].set(self._h())

        return self._outputs[index]

    # Creating input object
    def createInput(self, value=0):
        edge = Edge(value)
        self._inputs.append(edge)
        self._num_inputs += 1
        return self._num_inputs - 1  # Returning index

    # Creating output object
    def createOutput(self, value=0):
        edge = Edge(value)
        self._outputs.append(edge)
        self._num_outputs += 1
        return self._num_inputs - 1  # Returning index

    # New label name
    def setLabel(self, label="default_cell"):
        self._label = label
        return self._label

    # Describing cell
    def describe(self):
        """ Describing this cell """
        message = "Cell '{0}' has {1} inputs and {2} outputs."
        print(message.format(self._label, self._num_inputs, self._num_outputs))

        for i in self._inputs:
            print("input value={0} weight={1}".format(i.get(), i.getWeight()))

        return True

    # Clear
    def clear(self):
        """ Clearing cell's info """

        # Clearig properties
        self._inputs = []
        self._outputs = []
        self._features = []

        self._num_inputs = 0  # Total of inputs
        self._num_outputs = 0  # Total of outputs
        self._label = "default_label"  # Default label


# List of layers
# A layer has a bunch of cells
class Layers():
    """docstring for LayersHandler"""

    # Properties
    _cells = []  # List of cells
    _num_cells = 0  # Cells counter
    label = "default_layer"

    # Getting cell
    def cell(self, index):
        try:
            return self._cells[index]
        except Exception:
            return None

    # Ctor
    def __init__(
        self,
        num_cells=0,
        num_inputs=0,
        num_outputs=0,
        label="default_layer"
    ):
        # Properties
        self.label = label

        # Creating cells
        for i in xrange(0, num_cells):
            self.createCell(num_inputs, num_outputs, "cell_" + str(i))

    # Create Cell
    def createCell(self, num_inputs=2, num_outputs=1, label="default_cell"):
        """ Creating Cell """
        cell = Cells(num_inputs, num_outputs, label)
        self._cells.append(cell)
        self._num_cells += 1
        return self._num_cells - 1  # Returning index

    # Counting cells
    def countCells(self):
        return len(self._cells)

    # Describing layer
    def describe(self):
        """ Describing this layer """
        message = "Layer '{0}' has {1} cells."
        print(message.format(self.label, self.countCells()))
        return True


# Neural network abstraction
class NeuralNetwork(object):
    """docstring for NeuralNetwork"""

    # Properties
    _layers = []  # List of layers

    # Getting layer content
    def layer(self, index):
        return self._layers[index]

    # Ctor
    def __init__(
        self,
        num_layers=0,
        num_cells=1,
        num_inputs=2,
        num_outputs=1,
        label="Layer"
    ):
        # Creating layers
        for i in xrange(0, num_layers):
            self.createLayer(num_cells, num_inputs, num_outputs, label)

    # Creating a new Layer
    def createLayer(
        self,
        num_cells=0,
        num_inputs=0,
        num_outputs=0,
        label="default_layer"
    ):
        layer = Layers(num_cells, num_inputs, num_outputs, label)
        self._layers.append(layer)

    # Counting number of layers
    def countLayers(self):
        return len(self._layers)

    # Describing neural network
    def describe(self):
        message = "This neural network has {0} Layers."

        # Counting number of layers
        print(message.format(self.countLayers()))

        # Describing layers
        for layer in self._layers:
            layer.describe()

        return True


# Main
def main():
    print("--------------")
    print("Neural network")
    print("--------------\n")

    # Building network
    n = NeuralNetwork()  # New network

    # Layer
    n.createLayer(label="layer_0")  # Creating empty layer

    # Cell
    n.layer(0).createCell(3, 1, "main_cell")  # Creating cell

    # Setting inputs
    n.layer(0).cell(0).input(0).set(1)
    n.layer(0).cell(0).input(1).set(1)
    n.layer(0).cell(0).input(2).set(1)

    # Setting weights
    n.layer(0).cell(0).input(0).setWeight(-20)
    n.layer(0).cell(0).input(1).setWeight(10)
    n.layer(0).cell(0).input(2).setWeight(10)

    # Output
    # output = n.layer(0).cell(0).output().get()

    # Describing cell 0
    n.layer(0).cell(0).describe()

# Program
if __name__ == "__main__":
    main()
