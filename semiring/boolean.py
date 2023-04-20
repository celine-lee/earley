from __future__ import annotations

from semiring import Element, Semiring
from dataclasses import dataclass
import torch

from grammar.grammar import ProbabilisticGrammar, GrammarRule, Score

""" Recognition """


@dataclass
class BooleanElement(Element):
    """A boolean element is just true or false."""

    score: bool

    def mul(self, other: BooleanElement) -> BooleanElement:
        return BooleanElement(self.score and other.score)

    def add(self, other: BooleanElement) -> BooleanElement:
        return BooleanElement(self.score or other.score)

    def __eq__(self, other) -> bool:
        return self.score == other.score

    def get_score(self, from_element=None):
        return self.score


class BooleanSemiring(Semiring):
    def __init__(self, p_grammar):
        super().__init__(mul_id=BooleanElement(True), add_id=BooleanElement(False))
        self.root_element = BooleanElement(True)
        self.p_grammar = p_grammar

    def initialize_element_from_rule(self, rule: GrammarRule):
        assert self.p_grammar is not None
        return BooleanElement(bool(self.p_grammar.parameter[rule.uid]))
