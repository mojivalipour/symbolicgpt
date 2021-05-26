# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp

"""Module implementing the Module class that represents a single model, 
defined by its canonical expression string.
    
An object of Model acts as a container for various representations of the model,
including its expression, symbols, parameters, the parse trees that simplify to it,
and associated information and references. 
Class methods serve as an interfance to interact with the model.
The class is intended to be used as part of an equation discovery algorithm."""

class Model:
    """Class that represents a single model, defined by its canonical expression string.
    
    An object of Model acts as a container for various representations of the model,
    including its expression, symbols, parameters, the parse trees that simplify to it,
    and associated information and references. 
    Class methods serve as an interfance to interact with the model.
    The class is intended to be used as part of an equation discovery algorithm.
    
    Attributes:
        expr (SymPy expression): The canonical expression defining the model.
        sym_vars (list of Sympy symbols): The symbols appearing in expr that are to be interpreted as variables.
        sym_params (list of strings): Symbols appearing in expr that are to be interpreted as free constants.
        params (list of floats): The values for the parameters, initial or estimated.
        estimated (dict): Results of optimization. Required items:
            "x": solution of optimization, i.e. optimal parameter values (list of floats)
            "fun": value of optimization function, i.e. error of model (float)
        valid (boolean): True if parameters successfully estimated. 
            False if estimation has not been performed yet or if it was unsuccessful. 
        trees (dict): Tracks parse trees that simplify to expr. Keys are codes of parse trees, values are a list with:
            probability of parse tree (float)
            number of occurences during sampling (int)
        p (float): Total probability of model. Computed as sum of probabilities of parse trees.
        grammar (GeneratorGrammar): Grammar the produced the model. 
                In the future will likely be generalized to BaseExpressionGenerator and tracked for each parse tree.
        
    Methods:
        add_tree: Add a new parse tree to the parse tree dict and update the probabilities.
        set_estimated: Save results of parameter estimation and set model validity according to input.
        get_error: Return the model error if model valid or a dummy value if model not valid.
        lambdify: Produce callable function from symbolic expression and parameter values. 
        evaluate: Compute the value of the expression for given variable values and parameter values.
        full_expr: Produce symbolic expression with parameters substituted by their values.
    """
    
    def __init__(self, expr, code, p, grammar=None, params=[], sym_params=[], sym_vars = []):
        """Initialize a Model with the initial parse tree and information on the task.
        
        Arguments:
            expr (Sympy expression or string): Expression that defines the model.
            code (string): Parse tree code, expressed as string of integers, corresponding to the choice of 
                production rules when generating the expression. Allows the generator to replicate 
                the generation. Requires the originating grammar to be useful.
            p (float): Probability of initial parse tree.
            grammar (nltk.PCFG or GeneratorGrammar): Grammar that generates the parse trees for this model.
                In the future will likely be generalized to BaseExpressionGenerator and tracked for each parse tree.
            params (list of floats): (Initial) parameter values.
            sym_vars (list of Sympy symbols): The symbols appearing in expr that are to be interpreted as variables.
            sym_params (list of strings): Symbols appearing in expr that are to be interpreted as free constants.
            """
        
        self.grammar = grammar
        self.params = params
        
        if isinstance(expr, type("")):
            self.expr = sp.sympify(expr)
        else:
            self.expr = expr
     
        try:
            self.sym_params = sp.symbols(sym_params)
            if type(self.sym_params) != type((1,2)):
                if isinstance(sym_params, list):
                    self.sym_params = tuple(sym_params)
                elif isinstance(sym_params, (int, float, str)):
                    self.sym_params = (self.sym_params, )
                else:
                    print("Unknown type passed as sym_params input of Model."\
                          "Valid types: tuple or list of strings."\
                          "Example: ('C1', 'C2', 'C3').")
        except ValueError:
            print(expr, params, sym_params, sym_vars)
        self.sym_vars = sp.symbols(sym_vars)
        self.p = 0
        self.trees = {} #trees has form {"code":[p,n]}"
        
        if len(code)>0:
            self.add_tree(code, p)
        self.estimated = {}
        self.valid = False
        
    def add_tree (self, code, p):
        """Add a new parse tree to the model.
        
        Arguments:
            code (str): The parse tree code, expressed as a string of integers.
            p (float): Probability of parse tree.
        """
        if code in self.trees:
            self.trees[code][1] += 1
        else:
            self.trees[code] = [p,1]
            self.p += p
        
    def set_estimated(self, result, valid=True):
        """Store results of parameter estimation and set validity of model according to input.
        
        Arguments:
            result (dict): Results of parameter estimation. 
                Designed for use with methods, implemented in scipy.optimize, but works with any method.
                Required items:
                    "x": solution of optimization, i.e. optimal parameter values (list of floats)
                    "fun": value of optimization function, i.e. error of model (float).
            valid: True if the parameter estimation succeeded.
                Set as False if the optimization was unsuccessfull or the model was found to not fit 
                the requirements. For example, we might want to limit ED to models with 5 or fewer parameters
                due to computational time concerns. In this case the parameter estimator would refuse
                to fit the parameters and set valid = False. 
                Invalid models are typically excluded from post-analysis."""
        
        self.estimated = result
        self.valid = valid
        if valid:
            self.params = result["x"]
        
    def get_error(self, dummy=10**8):
        """Return model error if the model is valid, or dummy if the model is not valid.
        
        Arguments:
            dummy: Value to be returned if the parameter have not been estimated successfully.
            
        Returns:
            error of the model, as reported by set_estimated, or the dummy value.
        """
        if self.valid:
            return self.estimated["fun"]
        else:
            return dummy
        
    def set_params(self, params):
        self.params=params
        
    def lambdify (self, *params, arg="numpy"):
        """Produce a callable function from the symbolic expression and the parameter values.
        
        This function is required for the evaluate function. It relies on sympy.lambdify, which in turn 
            relies on eval. This makes the function somewhat problematic and can sometimes produce unexpected
            results. Syntactic errors in variable or parameter names will likely produce an error here.
        
        Arguments:
            arg (string): Passed on to sympy.lambdify. Defines the engine for the mathematical operations,
                that the symbolic operations are transformed into. Default: numpy.
                See sympy documentation for details.
                
        Returns:
            callable function that takes variable values as inputs and return the model value.
        """
        if not params:
            params = self.params
        return sp.lambdify(self.sym_vars, self.full_expr(*params), "numpy")
        # self.lamb_expr = sp.lambdify(self.sym_vars, self.expr.subs(list(zip(self.sym_params, params))), arg)
        # print(self.lamb_expr, "self.lamb_expr")
        # test = self.lamb_expr(np.array([1,2,3, 4]))
        # print(test, "test")
        # if type(test) != type(np.array([])):
        #     print("inside if, i.e. bool=True")
        #     self.lamb_expr = lambda inp: [test for i in range(len(inp))]
        # return self.lamb_expr

    def evaluate (self, points, *args):
        """Evaluate the model for given variable and parameter values.
        
        If possible, use this function when you want to do computations with the model.
        It relies on lambdify so it shares the same issues, but includes some safety checks.
        Example of use with stored parameter values:
            predictions = model.evaluate(X, *model.params)
        
        Arguments:
            points (numpy array): Input data, shaped N x M, where N is the number of samples and
                M the number of variables.
            args (list of floats): Parameter values.
            
        Returns:
            Numpy array of shape N x D, where N is the number of samples and D the number of output variables.
        """
        lamb_expr = sp.lambdify(self.sym_vars, self.full_expr(*args), "numpy")
        
        if type(points[0]) != type(np.array([1])):
            if type(lamb_expr(np.array([1,2,3]))) != type(np.array([1,2,3])):
                return np.ones(len(points))*lamb_expr(1)
            return lamb_expr(points)
        else:
#            if type(lamb_expr(np.array([np.array([1,2,3])]).T)) != type(np.array([1,2,3])):
            return lamb_expr(*points.T)
    
    def full_expr (self, *params):
        """Substitutes parameter symbols in the symbolic expression with given parameter values.
        
        Arguments:
            params (list of floats): Parameter values.
            
        Returns:
            sympy expression."""
        if type(self.sym_params) != type((1,2)):
            return self.expr.subs([[self.sym_params, params]])
        else:
            return self.expr.subs(list(zip(self.sym_params, params)))
        
    def get_full_expr(self):
        return self.full_expr(*self.params)
    
    def __str__(self):
        return str(self.expr)
    
    def __repr__(self):
        return str(self.expr)
    
    
    
if __name__ == '__main__':
    print("--- model.py test ---")
    from nltk import PCFG
    grammar_str = "S -> 'c' '*' 'x' [1.0]"
    grammar = PCFG.fromstring(grammar_str)
    parse_tree_code = "0"
    expression_str = "c*x"
    probability = 1.0
    symbols_params = ["c"]
    symbols_variables = ["x"]
    
    print("Create the model instance and print the model.")
    model = Model(expr = expression_str, 
                  grammar = grammar, 
                  code = parse_tree_code, 
                  p = probability,
                  sym_params = symbols_params,
                  sym_vars = symbols_variables)
    print(model)
    assert str(model) == expression_str
    
    print("Try to print the model error before it thas been estimated."\
          "The model returns the dummy value for an invalid model.")
    print(model.get_error())
    assert model.get_error() == 10**8
    
    print("Perform parameter estimation and add the results to the model."\
          "Then, print the model with the parameter values substituted.")
    result = {"x":[1.2], "fun":0.001}
    model.set_estimated(result)
    
    print(model.full_expr(*model.params))
    assert str(model.full_expr(*model.params)) == "1.2*x"
    
    print("Evaluate the model at points X.")
    X = np.reshape(np.linspace(0, 5, 2), (2, 1))
    y = model.evaluate(X, *model.params)
    print(y)
    assert isinstance(y, type(np.array([0])))
    assert sum((y - np.array([0, 6.0]))**2) < 1e-15
    

