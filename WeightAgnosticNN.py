# -*- coding: utf-8 -*-
"""
Project inspired by:
https://arxiv.org/pdf/1906.04358.pdf

The overall idea is to select the best network architecture for a given problem
- being agnostic on weights
- letting the network evolving in a similar idea as genetic algorithms

"""

import copy
import random

import numpy as np
import pandas as pd

from keras.models import Sequential 
from keras.layers import Dense

from keras.backend import clear_session


class Genome():
    """Class representing a Genome in the idea of evolutionary neural networks.
    This is the basic element, aimed to evolve.
    
    Attributes:
    ----------------
    - _inputs: number of inputs in the neural network
    - _outputs: number of outputs of the neural network
    - ACTIVATIONS: dictionary of activations
    - model: the neural network
    - _complexity: the number of parameters of the model
        
    Methods (main ones):
    ----------------
    _initialize: initialize the genome
    _set_weights: set all weights for all matrix to a single value
    _mutate: perform one mutation step
    predict: run the fed forward process through the network, fixing weights
    
    """
    
    # Potential activations, directly taken from Keras
    ACTIVATIONS = {0: 'sigmoid',
                   1: 'tanh',
                   2: 'relu',
                   3: 'softplus',
                   4: 'softsign',
                   5: 'selu',
                   6: 'elu',
                   7: 'exponential'}
    
    def __init__(self, inputs, outputs):
        """Initialize the genome
        
        :param inputs: (int) the feature dimension of the explanatory variable
        :param outputs: (int) the feature dimension of the target variable
        """
        
        self._inputs = inputs
        self._outputs = outputs
        
        # Initialize neural network
        self.model = []
        self._complexity = 0
        
        # Verify the dictionary
        self._verify_dictionary_intergrity()
        
    def __copy__(self):
        """Implement a copy method"""
        
        g = Genome(self._inputs, self._outputs)
        g.model = self.model
        g._complexity = self._complexity
        
        return g
        
    def _verify_dictionary_intergrity(self):
        """Function verifying there is no issue in the activation dictionary
        at genome instanciation"""
        
        for i in range(len(self.ACTIVATIONS)):
            if not i in self.ACTIVATIONS.keys():
                print('The dictionary containing activations do not respect its lentgh')
                return -1
        
    def _set_model(self, model):
        """Assign the model and its complexity (number of parameters) to the
        Genome
        
        :param: model: (keras object) a neural network
        """
        self.model = model
        self._complexity = model.count_params()
        
    def _initialize(self):
        """Initialize the neural network with the minimal size. It is composed
        by:
        - one hidden layer, with the number of inputs being initially defined
        - the output is a layer with the number of outputs being also initially
          defined
        """
          
        # Create the model
        model = Sequential()
        model.add(Dense(1, input_shape = (self._inputs,), activation=self.select_activation()))
        model.add(Dense(self._outputs, activation=self.select_activation()))
        
        # Assign the model to the genome and computes its complexity
        self._set_model(model)
        
        return self
        
    def _set_weights(self, weight_value):
        """Function setting all weights of all layer to a single value
        
        :params weight_value: (float) value to be assigned to all weights
        """
        
        for layer in self.model.layers:
            weights_list = layer.get_weights()
            weights_list = [np.ones(shape = x.shape) * weight_value for x in weights_list]
            layer.set_weights(weights_list)

    def select_activation(self):
        """Function randomly selecting the activation function"""
        return self.ACTIVATIONS[random.choice(list(range(len(self.ACTIVATIONS))))]

    def _mutate(self, proba_unit_shift, proba_unit_mutate, max_unit_generated, 
                proba_change_activation, proba_add_layer):
        """Function to mutate a Genome, with probabilities thresholds defined 
        as hyperparameters:
            
        :param proba_unit_shift: (float) probability to add / remove one unit in a layer
        :param proba_unit_mutate: (float) probability to change the number of units in a layer
        :param max_unit_generated: (int) maximum number of unit a layer can have
        :param proba_change_activation: (float) probability to change the activation, to select
          a random one
        :param proba_add_layer: (float) probability to add a new layer
        """
        
        # To keep track of the first layer
        n_incr = 0
        
        # Initialize the new model
        model = Sequential()
        
        # Generate the random numbers
        rand1, rand2, rand3 = random.random(), random.random(), random.random()
        
        # Loop through existing layers
        for layer in self.model.layers[:-1]:
            
            # Get characteristics
            unit, activation = layer.units, layer.activation
            
            # Step 1: Number of units            
            # Case 1: shift number of units
            if rand1 < proba_unit_shift:
                unit += random.choice([-1, 1])
                if unit < 1:
                    unit = 1
                    
            # Case 2: mutate number of units
            elif proba_unit_shift <= rand1 < proba_unit_shift + proba_unit_mutate:
                unit = random.randint(1, max_unit_generated)
            
            # Step 2: Activation          
            if rand2 < proba_change_activation:
                activation = self.select_activation()
            
            # Depending if we are first layer or not, as the input shape might need to be defined
            if n_incr != 0:
                model.add(Dense(unit, activation = activation)) 
            else:
                model.add(Dense(unit, input_shape = (self._inputs,), activation = activation))
            
            n_incr += 1
            
        # Step 3: Add a other layer
        if rand3 < proba_add_layer:
            unit = random.randint(1, max_unit_generated)
            activation = self.select_activation()
            model.add(Dense(unit, activation = activation))   
            
        # Step 4: Add the final layer
        model.add(Dense(self._outputs, activation = self.model.layers[-1].activation) )   
        
        # Assign the model to the genome and computes its complexity
        self._set_model(model)
        
        return self
        
    def predict(self, x, weight_values):
        """Runs the predict function for a list of values to be assigned to the
        neural network weights
        
        :param x: [m inputs] numpy matrix of data
        :param weight_values: (list) list with the values to be assigned 
        sucessively
        """
        
        # Initialize the output
        output = []
        
        for weight in weight_values:
            self._set_weights(weight)
            fcst = self.model.predict(x)
            output.append(fcst)
            
        return output


class Population():
    """Class encapsulating the population with the different attributes and 
    methods to run the algorithm, by making the genomes evolving.

    
    Attributes:
    ----------------
    - _size: population size
    - _proba_unit_shift: probability to add / remove one unit in a layer
    - _proba_unit_mutate: probability to change the number of units in a layer
    - _max_unit_generated: maximum number of unit a layer can have
    - _proba_change_activation: probability to change the activation, to select
      a random one
    - _proba_add_layer: probability to add a new layer
    - _proba_use_complexity: probability to use the complexity as second factor
    while ranking the model (otherwise it uses the 'best' metrics)
    - _to_keep_random: percentage of random models to keep, for diversity
    - population: the Genome population
    - best_loss: Track the best loss of the population model
    
    - COEFFICIENTS: coefficients to be used to run the feed forward run of
    each Genome
    
    Methods (main ones):
    ----------------
    - _generate_population: initialize the population
    - determine_performance: computes the metrics that matters for a model
    - run_epoch: run a single epoch
    - run: run the full evolution approch
    
    """
    
    COEFFICIENTS = [-2, -1, -0.5, 0.5, 1, 2]
    
    def __init__(self, size, proba_unit_shift, proba_unit_mutate, max_unit_generated,
                 proba_change_activation, proba_add_layer, proba_use_complexity,
                 to_keep_random):
        """Initialize the attributes

        :param size: (int)
        :param proba_unit_shift: (float)
        :param proba_unit_mutate: (float)
        :param max_unit_generated: (int)
        :param proba_change_activation:(float)
        :param proba_add_layer: (float)
        :param proba_use_complexity: (float)  
        :param to_keep_random: (float)
        """

        # Constants
        self._size = size
        self._proba_unit_shift = proba_unit_shift
        self._proba_unit_mutate = proba_unit_mutate
        self._max_unit_generated = max_unit_generated
        self._proba_change_activation = proba_change_activation
        self._proba_add_layer = proba_add_layer
        self._proba_use_complexity = proba_use_complexity
        self._to_keep_random = to_keep_random
        
        # Populations
        self.population = []
        self.best_loss = []

    def _generate_population(self, inputs, outputs):
        """Generate and initialize the population
        
        :param inputs: (int) the feature dimension of the explanatory variable
        :param outputs: (int) the feature dimension of the target variable
        """ 
        self.population = [Genome(inputs, outputs)._initialize() for _ in range(self._size)]
         
    def _mutate_population(self):
        """Make one mutation step"""
        
        # Create new individuals, so need to copy the ones currently in memory
        population_mutated = [copy.copy(g) for g in self.population]
        
        return [g._mutate(self._proba_unit_shift,
                          self._proba_unit_mutate,
                          self._max_unit_generated,
                          self._proba_change_activation,
                          self._proba_add_layer) for g in population_mutated]
    
    def _combine_populations(self, population_mutated):
        """Add the initial and the mutated population together
        
        :param population_mutated: (list) list of Genomes
        """
        
        population = self.population
        population.extend(population_mutated)
        self.population = population
        
    def _get_best_metric(self, weights_metrics, mode):
        """Get the best metric, according to the mode
        
        :param weights_metrics: (list) list containing the losses computed for
        the differents coefficients
        :param mode: (string) determines if looking for the minimal or maximal
        value of losses. Needs to be 'min' or 'max'
        """
                
        if mode == 'min':
            metric_best = np.min(weights_metrics)
        elif mode == 'max':
            metric_best = np.max(weights_metrics)
            
        return metric_best
    
    def determine_performance(self, y, y_hat, g, loss_function, mode):
        """For a given model, return the 3 metrics used for comparison:
        - the mean performance of the network structure across all weights
        - the best performance of the network for a weight
        - the complexity of the network
        
        :param y: [m output] numpy matrix, this is the target variable
        :param y_hat: [m output] numpy matrix, this is the target variable 
        forecasted by the model
        :param g: (Genome) the Genome
        :param loss_function: (function) the function used to compute the loss
        between y and y_hat
        :param mode: (string) to determine the best metric across the weights 
        calculations, need to be 'min' or 'max'
        
        """
        
        # Initialize individual weight metrics into a list 
        weights_metrics = []
        for forecast in y_hat:
            weights_metrics.append(loss_function(y, forecast))
            
        # Computes the metrics for ranking
        metric_mean = np.mean(weights_metrics)
        metric_best = self._get_best_metric(weights_metrics, mode)
        complexity = g._complexity
        
        return metric_mean, metric_best, complexity
            
    def run_epoch(self, X, y, loss_function, mode):
        """Make a pass through for one epoch:
        - For each genome, for each coefficient run a pass through
        - Compare the results with the target results
        - Store the criterion in the dataframe
        
        :param X: [m inputs] numpy matrix with the explanatory variable
        :param y: [m outputs] numpy matrix with the target variable
        :param loss_function: (function) the function used to compute the loss
        between y and y_hat
        :param mode: (string) to determine the best metric across the weights 
        calculations, need to be 'min' or 'max'        
        
        """
        
        i = 0
        output = pd.DataFrame(index=range(len(self.population)), 
                              columns=['mean', 'best', 'complexity'])
        
        for g in self.population:
            
            # Make the prediction and then derive the criterions
            y_hat = g.predict(X, self.COEFFICIENTS)
            m, b, c = self.determine_performance(y, y_hat, g, loss_function, mode)
            
            output.at[i, 'mean'] = m
            output.at[i, 'best'] = b
            output.at[i, 'complexity'] = c
            i += 1
        
        return output
    
    def run(self, X, y, loss_function, epochs, mode='min'):
        """Run the algorithm evolution
        
        :param X: [m inputs] numpy matrix with the explanatory variable
        :param y: [m outputs] numpy matrix with the target variable
        :param loss_function: (function) the function used to compute the loss
        between y and y_hat
        :param epochs: (int) number of epochs to go through
        :param mode: (string) to determine the best metric across the weights 
        calculations, need to be 'min' or 'max'   
        
        """
    
        # Check 'mode' has been properly inputed
        assert mode in ['min', 'max'], "The 'mode' input has not been properly inputed"
    
        # First step: initialize and first mutation
        self._generate_population(X.shape[1], y.shape[1])
        
        for epoch in range(epochs):
            print('-- epoch {0}/{1}'.format(epoch, epochs))
            
            # Aggregate the full population and compute the metrics
            population_mutated = self._mutate_population()
            self._combine_populations(population_mutated)
            
            metrics = self.run_epoch(X, y, loss_function, mode)
        
            # Sort the metrics
            rand = random.random()
            if rand < self._proba_use_complexity:
                criterion = ['mean', 'complexity']
            else:
                criterion = ['mean', 'best']
                    
            metrics = metrics.sort_values(by=criterion, ascending=True)
            
            # Keep the best ones plus some random, for diversity in the evolution
            threshold = int(self._size * (1 - self._to_keep_random))
            to_keep = metrics.iloc[:threshold]
            to_add = random.choices(list(metrics.iloc[threshold:].index), k=self._size-threshold)
            
            # Track the best loss of this epoch
            self.best_loss.append(to_keep.iloc[0,0])
            
            print('best loss for this epoch is {0}'.format(to_keep.iloc[0,0]))
            print('loss average for the 5 best models is {0}'.format(np.mean(to_keep.iloc[:5,0])))
            print('max complexity tested {0}'.format(metrics['complexity'].max()))
            
            population = []
            for idx in to_keep.index:
                population.append(self.population[idx])
                
            for idx in to_add:
                population.append(self.population[idx])
                
            self.population = population
            
            clear_session()


if __name__ == '__main__':
    
    # Random example
    
    # Initialize random variables
    X = np.random.randint(-10,10,size=(1000,10))
    y = np.random.randint(-10,10,size=(1000,1))
    
    # Initialize the algorithm
    pop = Population(40,0.5,0.5,100,0.5,0.5,0.8, 0.5)

    # Define a loss function, here the mean squared errors
    def loss_function(y, y_hat):
        return np.sqrt(np.sum((y - y_hat)**2))
    
    pop.run(X, y, loss_function, 300, mode='min')
    

        
        
        