""" 
This directory implements a bunch of models and associated functions (in utils file). The models are 
indepedent bayesian nets, chow liu trees, cutset networks and their approximations. The file main.py 
should be used to try out training the  different models on the 10 datasets provided. 
Results are written to the ../results/ folder, through pickled dictionaries which track the 
train, test, and val log likelihoods along with the runtime for training the model.
"""

__author__ = "Ninad Khargonkar"
