# -*- coding: utf-8 -*-

from ProGED.model import Model
from ProGED.model_box import ModelBox, symbolic_difference
from ProGED.generators.grammar import GeneratorGrammar
from ProGED.generators.grammar_construction import grammar_from_template
from ProGED.parameter_estimation import fit_models
from ProGED.task import EDTask
from ProGED.equation_discoverer import EqDisco

__version__ = 0.8