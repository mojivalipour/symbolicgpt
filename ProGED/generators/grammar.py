# -*- coding: utf-8 -*-

import numpy as np
from nltk import PCFG
from nltk.grammar import Nonterminal, ProbabilisticProduction

from ProGED.generators.base_generator import BaseExpressionGenerator

class GeneratorGrammar (BaseExpressionGenerator):
    def __init__ (self, grammar):
        self.generator_type = "PCFG"
        self.coverage_dict = {}
        self.count_dict = {}
    
        if isinstance(grammar, str):
            self.grammar = PCFG.fromstring(grammar)
        elif isinstance(grammar, type(PCFG.fromstring("S -> 'x' [1]"))):
            self.grammar = grammar
        else:
            raise TypeError ("Unknown grammar specification. \n"\
                             "Expected: string or nltk.grammar.PCFG object.\n"\
                             "Input: " + str(grammar))
                
        self.start_symbol = self.grammar.start()
    
    def generate_one (self):
        return generate_sample(self.grammar, items=[self.start_symbol])
    
    def code_to_expression (self, code):
        return code_to_sample(code, self.grammar, items=[self.start_symbol])

    def count_trees(self, start, height):
        """Counts all trees of height <= height."""
        if not isinstance(start, Nonterminal):
            return 1
        elif height == 0:
            return 0
        else:
            counter = 0
            prods = self.grammar.productions(lhs=start)
            for prod in prods:
                combinations = 1
                for symbol in prod.rhs():
                    symbol_height_key = "("+str(symbol)+","+str(height-1)+")" 
                    if not symbol_height_key in self.count_dict:
                        self.count_dict[symbol_height_key] = self.count_trees(symbol, height-1)
                    combinations *= self.count_dict[symbol_height_key]
                counter += combinations
            return counter

    def count_coverage(self, start, height):
        """Counts total probability of all parse trees of height <= height."""
        if not isinstance(start, Nonterminal):
            return 1
        elif height == 0:
            return 0
        else:
            coverage = 0
            prods = self.grammar.productions(lhs=start)
            for prod in prods:
                subprobabs = prod.prob()
                for symbol in prod.rhs():
                    subprobabs *= self.count_coverage(symbol, height-1)
                coverage += subprobabs
            return coverage

    def count_coverage_external(self, start, height):
        """Counts coverage fast using external (objective) cache."""
        if height == 0:
            return 0
        coverage = 0
        prods = self.grammar.productions(lhs=start)
        for prod in prods:
            subprobabs = prod.prob()
            for symbol in prod.rhs():
                if not isinstance(symbol, Nonterminal):
                    continue
                elif (height-1) == 0:
                    subprobabs = 0
                    break
                else:
                    if "("+str(symbol)+","+str(height-1)+")" in self.coverage_dict:
                        subprobabs *= self.coverage_dict["("+str(symbol)+","+str(height-1)+")"]
                    else:
                        newprob = self.count_coverage_external(symbol, height-1)
                        self.coverage_dict["("+str(symbol)+","+str(height-1)+")"] = newprob
                        subprobabs *= newprob
            coverage += subprobabs
        return coverage

    def list_coverages(self, height, tol=10**(-17),
                            min_height=100, verbosity=0):
        """Counts coverage of maximal height using cache(dictionary).

        Input:
            height - maximal height of parse trees of which the
                coverage is calculated of.
            tol - tolerance as a stopping condition. If change
                is smaller than the input tolerance, then it stops.
            min_height - overrides tolerance stopping condition and
                calculates coverage of all heights <= min_height. It
                also determines for how many previous steps the change
                is measured, i.e. for levels (height-1 - min_height/2).
            verbosity - if set to > 0, it prints stopping probability
                change, height and input tolerance.
        Output:
            Dictionary with nonterminals as keys and coverages of all
            parse trees with root in given key nonterminal and their
            heights at most the input height as their values.
        """
        nonterminals = list(set([prod.lhs() for prod
                                in self.grammar.productions()]))
        if height == 0:
            return {A: 0 for A in nonterminals}
        probs_dict = {}
        for A in nonterminals:      # height = 0:
            probs_dict[(A, 0)] = 0
        min_height = max(min_height, 2)  # to avoid int(min_height/2)=0
        for level in range(1, height+1):    # height > 0:
            if level > min_height:  # Do `min_height` levels without stopping.
                # Measure change from last min_height/2 levels:
                change = max(abs(probs_dict[(A, level-1)]
                                 - probs_dict[(A, level-int(min_height/2))])
                                 for A in nonterminals)
                if change < tol:
                    if verbosity > 0:
                        print(change, level, tol, "change of probability")
                    return {A: probs_dict[(A, level-1)] for A in nonterminals}
            for A in nonterminals:
                coverage = 0
                prods = self.grammar.productions(lhs=A)
                for prod in prods:
                    subprobabs = prod.prob()
                    for symbol in prod.rhs():
                        if not isinstance(symbol, Nonterminal):
                            continue  # or subprobabs = 1
                        else:
                            subprobabs *= probs_dict[(symbol, level-1)]
                    coverage += subprobabs
                probs_dict[(A, level)] = coverage
        if verbosity > 0:
            print("The input height %d was reached. " % height
                +"Bigger height is needed for better precision.")
        return {A: probs_dict[(A, height)] for A in nonterminals}

    def renormalize(self, height=10**4, tol=10**(-17), min_height=100):
        """Return renormalized grammar. 
        
        Raise ValueError if for at least one nonterminal, its coverage
        equals zero.
        Input:
            height - maximal height of parse trees of which the
                coverage is calculated of.
            tol - tolerance as a stopping condition. If change
                is smaller than the input tolerance, then it stops.
            min_height - overrides tolerance stopping condition and
                calculates coverage of all heights <= min_height. It
                also determines for how many previous steps the change
                is measured, i.e. for levels (height-1 - min_height/2).
            verbosity - if set to > 0, it prints stopping probability
                change, height and input tolerance.
        """
        coverages_dict = self.list_coverages(height, tol, min_height)
        if min(coverages_dict[A] for A in coverages_dict) < tol:  # input tol
            raise ValueError("Not all coverages are positive, so"
                            + " renormalization cannot be performed since zero"
                            + " division.")
        def chi(prod, coverages_dict):
            """Renormalizes production probability p^~ as in Chi paper(22)."""
            subprobabs = prod.prob()
            for symbol in prod.rhs():
                if not isinstance(symbol, Nonterminal):
                    continue  # or subprobabs = 1
                else:
                    subprobabs *= coverages_dict[symbol]
            return subprobabs/coverages_dict[prod.lhs()]
        prods = [ProbabilisticProduction(prod.lhs(), prod.rhs(),
                                        prob=chi(prod, coverages_dict))
                for prod in self.grammar.productions()]
        return PCFG(self.grammar.start(), prods)

    def __str__ (self):
        return str(self.grammar)
    
    def __repr__ (self):
        return str(self.grammar)
    
    

def generate_sample_alternative(grammar, start):
    """Alternative implementation of generate_sample. Just for example."""
    if not isinstance(start, Nonterminal):
        return [start], 1, ""
    else:
        prods = grammar.productions(lhs=start)
        probs = [p.prob() for p in prods]
        prod_i = np.random.choice(list(range(len(prods))), p = probs)
        frags = []
        probab = probs[prod_i]
        code = str(prod_i)
        for symbol in prods[prod_i].rhs():
            frag, p, h = generate_sample_alternative(grammar, symbol)
            frags += frag
            probab *= p
            code += h
        return frags, probab, code
    
def generate_sample(grammar, items=[Nonterminal("S")]):
    """Samples PCFG once. 
    Input:
        grammar - PCFG object from NLTK library
        items - list containing start symbol as Nonterminal object. Default: [Nonterminal("S")]
    Output:
        frags - sampled string in list form. Call "".join(frags) to get string.
        probab - parse tree probability
        code - parse tree encoding. Use code_to_sample to recover the expression and productions.
    """
#    print(items)
    frags = []
    probab = 1
    code = ""
    if len(items) == 1:
        if isinstance(items[0], Nonterminal):
            prods = grammar.productions(lhs=items[0])
            probs = [p.prob() for p in prods]
            prod_i = np.random.choice(list(range(len(prods))), p = probs)
            frag, p, h = generate_sample(grammar, prods[prod_i].rhs())
            frags += frag
            probab *= p * probs[prod_i]
            code += str(prod_i) + h
        else:
            frags += [items[0]]
    else:
        for item in items:
            frag, p, h = generate_sample(grammar, [item])
            frags += frag
            probab *= p
            code += h
    return frags, probab, code

def code_to_sample (code, grammar, items=[Nonterminal("S")]):
    """Reconstructs expression and productions from parse tree encoding.
    Input:
        code - parse tree encoding in string format, as returned by generate sample
        grammar - PCFG object that was used to generate the code
        items - list containing start symbol for the grammar. Default: [Nonterminal("S")]
    Output:
        frags - expression in list form. Call "".join(frags) to get string.
        productions - list of used productions in string form. The parse tree is ordered top to bottom, left to right.
        code0 - auxilary variable, used by the recursive nature of the function. Should be an empty string. If not, something went wrong."""
    code0 = code
    frags = []
    productions=[]
    if len(items) == 1:
        if isinstance(items[0], Nonterminal):
            prods = grammar.productions(lhs=items[0])
            prod = prods[int(code0[0])]
            productions += [prod]
            frag, productions_child, code0 = code_to_sample(code0[1:], grammar, prod.rhs())
            frags += frag
            productions += productions_child
        else:
            frags += [items[0]]
    else:
        for item in items:
            frag, productions_child, code0 = code_to_sample (code0, grammar, [item])
            frags += frag
            productions += productions_child
    #print(frags, code0)
    return frags, productions, code0

def sample_improper_grammar (grammar):
    try:
        return generate_sample(grammar)
    except RecursionError:
        return []
    
if __name__ == "__main__":
    print("--- generators.grammar.py test ---")
    np.random.seed(0)
    grammar = GeneratorGrammar("E -> 'x' [0.7] | E '*' 'x' [0.3]")
    for i in range(5):
        np.random.seed(i)
        f, p, c = grammar.generate_one()
        print((f, p, c))
        np.random.seed(i)
        print(generate_sample_alternative(grammar.grammar, grammar.start_symbol))
        print(code_to_sample(c, grammar.grammar, [grammar.start_symbol]))
        print(grammar.count_trees(grammar.start_symbol,i))
        print(grammar.count_coverage(grammar.start_symbol,i))
        print(grammar.list_coverages(i)[grammar.start_symbol])
    print("\n-- testing different grammars: --\n")
    pgram0 = GeneratorGrammar("""
        S -> 'a' [0.3]
        S -> 'b' [0.7]
    """)
    pgram1 = GeneratorGrammar("""
        S -> A B [0.8]
        S -> 's' [0.2]
        A -> 'a' [1]
        B -> 'b' [0.3]
        B -> C D [0.7]
        C -> 'c' [1]
        D -> 'd' [1]
    """)
    pgramSS = GeneratorGrammar("""
        S -> S S [0.3]
        S -> 'a' [0.7]  
    """)
    def pgramSSparam(p=0.3):
        return GeneratorGrammar(f"""
                S -> S S [{p}]
                S -> 'a' [{1-p}]  
    """)
    pgrama = GeneratorGrammar("""
        S -> A B [0.7]
        S -> 'a' [0.1]
        S -> 'b' [0.1]
        S -> 'c' [0.1]
        A -> A1 A2 [0.3]
        A -> 'aq' [0.5]
        A -> 'bq' [0.2]
        B -> 'aw' [0.1]
        B -> 'bw' [0.9]
        A2 -> 'ak' [0.4]
        A2 -> 'bk' [0.6]
        A1 -> A11 A12 [1]
        A11 -> 'ar' [0.8]
        A11 -> 'br' [0.2]
        A12 -> 'af' [0.3]
        A12 -> 'bf' [0.7]
    """)
    pgramw = GeneratorGrammar("""
    S -> A B A2 A1 B2 B1 [0.7]
    A -> A1 A2 [0.3]
    B -> B1 B2 [0.1]
    S -> 'a' [0.1]
    S -> 'b' [0.1]
    S -> 'c' [0.1]
    A -> 'aq' [0.5]
    A -> 'bq' [0.2]
    B -> 'aw' [0.1]
    B -> 'bw' [0.8]
    A2 -> 'ak' [0.4]
    A2 -> 'bk' [0.6]
    A1 -> 'ar' [0.8]
    A1 -> 'br' [0.2]
    B2 -> 'af' [1]
    B1 -> 'bf' [1]
    """)
    pgramCounterExample = GeneratorGrammar("""
        A -> S 'c' [0.7]
        A -> 'b' [0.3]
        S -> S S [0.8]
        S -> 'a' [0.2]
    """)
    from time import time
    t1=0
    def display_time(t1): t2 = time(); print(10**(-3)*int((t2-t1)*10**3), "= seconds consumed"); return t2
    height = 10**1+5
    p=0.9
    for gramm in [grammar, pgram0, pgram1, pgrama, pgramw, pgramSS,
                    pgramCounterExample, pgramSSparam(p) ]:
        print(f"\nFor grammar:\n {gramm}")
        for i in range(height, height+1):
        # for i in range(0, 5):
            t2=display_time(t1); t1=t2
            print(gramm.count_trees(gramm.start_symbol,i), f" = count trees of height <= {i}")
            # print(gramm.count_coverage(gramm.start_symbol,i), f" = coverage(start,{i}) of height <= {i}")
            # t2=display_time(t1); t1=t2;
            b = gramm.count_coverage_external(gramm.start_symbol,i)
            print(b, f" = coverage(start,{i}) of height <= {i}")
            t2=display_time(t1); t1=t2;
            c = gramm.list_coverages(i, tol=10**(-17), min_height=100,
                 verbosity=1)[gramm.grammar.start()] 
            print(c, f" = list_coverages({i})[start] of height <= {i}")
            # print(gramm.list_coverages(i, tol=10**(-17), min_height=100,
            #     verbosity=1)[gramm.grammar.start()],
            #     f" = coverage = list_coverages({i})[start] of height <= {i}")
            t2=display_time(t1); t1=t2
            if not (b == c): raise ValueError("Coverages do not match!!!!")
            gramm.grammar = gramm.renormalize()
            print("Renormalized grammar:\n %s" % gramm)
            print(gramm.list_coverages(i), " = renormalized coverages")
    print(f"Chi says: limit probablity = 1/p - 1, i.e. p={p} => prob={1/p-1}")
