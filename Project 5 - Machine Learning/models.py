import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        converged = False

        # x is set of input nodes
        # y is output from corresponding x

        while not converged:
            converged = True

            for input,output in dataset.iterate_once(1):
                predictionOutput = self.get_prediction(input)
                actualOutput = nn.as_scalar(output)
                
                if actualOutput != predictionOutput:
                    converged = False
                    self.get_weights().update(input, actualOutput)

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batchSize = 200
        self.learningRate = 0.05

        self.hiddenLayerW1 = nn.Parameter(1, 512)
        self.hiddenLayerB1 = nn.Parameter(1, 512)

        self.outputLayerW = nn.Parameter(512, 1)
        self.outputLayerB = nn.Parameter(1, 1)

        self.layerList = [self.hiddenLayerW1, self.hiddenLayerB1, self.outputLayerW, self.outputLayerB]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        firstLayer = nn.ReLU(nn.AddBias(nn.Linear(x, self.hiddenLayerW1), self.hiddenLayerB1))
        outputLayer = nn.AddBias(nn.Linear(firstLayer, self.outputLayerW), self.outputLayerB)

        return outputLayer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        predictedValue = self.run(x)
        squareLoss = nn.SquareLoss(predictedValue, y)

        return squareLoss

    def train(self, dataset):
        """
        Trains the model.
        """
        # as stated in the problem, aiming to make loss equal or better than 0.02
        lossValue = float('inf')

        while lossValue > 0.02:
            for input,output in dataset.iterate_once(self.batchSize):
                gradients = nn.gradients(self.get_loss(input, output), self.layerList)

                for i in range(len(self.layerList)):
                    self.layerList[i].update(gradients[i], -self.learningRate)

                lossValue = nn.as_scalar(self.get_loss(input, output))

                

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batchSize = 100
        self.learningRate = 0.8

        self.hiddenLayerW1 = nn.Parameter(784, 200)
        self.hiddenLayerB1 = nn.Parameter(1, 200)

        self.outputLayerW = nn.Parameter(200, 10)
        self.outputLayerB = nn.Parameter(1, 10)

        self.layerList = [self.hiddenLayerW1, self.hiddenLayerB1, self.outputLayerW, self.outputLayerB]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        firstLayer = nn.ReLU(nn.AddBias(nn.Linear(x, self.hiddenLayerW1), self.hiddenLayerB1))
        outputLayer = nn.AddBias(nn.Linear(firstLayer, self.outputLayerW), self.outputLayerB)

        return outputLayer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        predictedValue = self.run(x)
        softMaxLoss = nn.SoftmaxLoss(predictedValue, y)

        return softMaxLoss

    def train(self, dataset):
        """
        Trains the model.
        """
        while dataset.get_validation_accuracy() < 0.98:
            for input,output in dataset.iterate_once(self.batchSize):
                gradients = nn.gradients(self.get_loss(input, output), self.layerList)

                for i in range(len(self.layerList)):
                    self.layerList[i].update(gradients[i], -self.learningRate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.batchSize = 20
        self.learningRate = 0.1

        self.initialLayerW = nn.Parameter(self.num_chars, 200)
        self.initialLayerB = nn.Parameter(1, 200)

        self.hiddenLayerW1 = nn.Parameter(200, 200)
        self.hiddenLayerB1 = nn.Parameter(1, 200)

        self.outputLayerW = nn.Parameter(200, len(self.languages))
        self.outputLayerB = nn.Parameter(1, len(self.languages))

        self.layerList = [self.initialLayerW, self.initialLayerB, self.hiddenLayerW1, self.hiddenLayerB1, self.outputLayerW, self.outputLayerB]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        continuousLayer = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.initialLayerW), self.initialLayerB))

        for x in xs[1:]:
            currentLayer = nn.ReLU(nn.AddBias(nn.Linear(x, self.initialLayerW), self.initialLayerB))
            recurrenceLayer = nn.ReLU(nn.AddBias(nn.Linear(continuousLayer, self.hiddenLayerW1), self.hiddenLayerB1))
            continuousLayer = nn.Add(currentLayer, recurrenceLayer)

        outputLayer = nn.AddBias(nn.Linear(continuousLayer, self.outputLayerW), self.outputLayerB)

        return outputLayer

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        predictedValue = self.run(xs)
        softMaxLoss = nn.SoftmaxLoss(predictedValue, y)

        return softMaxLoss

    def train(self, dataset):
        """
        Trains the model.
        """
        while dataset.get_validation_accuracy() < 0.87:
            for input,output in dataset.iterate_once(self.batchSize):
                gradients = nn.gradients(self.get_loss(input, output), self.layerList)

                for i in range(len(self.layerList)):
                    self.layerList[i].update(gradients[i], -self.learningRate)
