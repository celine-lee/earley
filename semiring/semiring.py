from __future__ import annotations
from typing import NewType, Any
from dataclasses import dataclass
import torch
from grammar.grammar import GrammarRule, ProbabilisticGrammar, Score

"""
Semiring parsing:
- semiring: defines a set of values over which multiplication, addition, the multiplicative identity, and the additive identity are defined.
- item-based description: an item defines a derivation uniquely
- grammar: (nonterminals, terminals, rules, root nonterminal)

Usage:
chart[item] = semiring element
"""

""" ITEM-BASED DESCRIPTION """


@dataclass
class Item:
    pass


""" SEMIRING """


@dataclass
class Element:
    def mul(self, other):
        raise NotImplementedError

    def add(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError

    def get_score(self):
        raise NotImplementedError



class Semiring:
    def __init__(self, mul_id: Element, add_id: Element):
        assert (
            add_id.mul(mul_id) == add_id
        ), f"Multiplicative identity does not hold: {add_id} mul {mul_id} is {add_id.mul(mul_id)}, expected {add_id}"
        assert (
            add_id.add(mul_id) == mul_id
        ), f"Additive identity does not hold: {add_id.add(mul_id)} =/= {mul_id}"
        assert (
            mul_id.add(add_id) == mul_id
        ), f"Additive identity does not hold: {mul_id.add(add_id)} =/= {mul_id}"
        self.mul_id = mul_id
        self.add_id = add_id
        self.p_grammar: ProbabilisticGrammar | None = None
        self.root_element: Element | None = None

    def initialize_element_from_rule(self, rule: GrammarRule):
        raise NotImplementedError


class Chart:
    pass
