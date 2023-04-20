from __future__ import annotations

from dataclasses import dataclass
import torch
import numpy as np

from semiring import Element, Semiring
from grammar.grammar import (
    Score,
    ProbabilisticGrammar,
    GrammarRule,
)

""" Inside < R_0^inf , + , * , 0 , 1 > """

########################## SERIAL PROCESSING ##########################


@dataclass
class InsideElement(Element):
    """An inside element is a real float from 0 to infinity representation in semiring parsing."""

    score: torch.Tensor  # R_0^inf, but for log is R_-inf^inf

    def mul(self, other: InsideElement) -> InsideElement:
        # multiply score, but add logprobs.
        return InsideElement(self.score + other.score)

    def add(self, other: InsideElement) -> InsideElement:
        # add scores, but logaddexp logprobs
        return InsideElement(np.logaddexp(self.score, other.score))

    def __str__(self):
        return f"Score: {self.score}, {self.score.requires_grad}"

    def __eq__(self, other) -> bool:
        return np.isclose(self.score, other.score)

    def get_score(self):
        return self.score

    def __hash__(self):
        return hash(self.score)


class InsideSemiring(Semiring):
    def __init__(self, p_grammar: ProbabilisticGrammar):
        super().__init__(
            mul_id=InsideElement(Score(np.log(1.0))),
            add_id=InsideElement(Score(np.log(1e-10))),
        )
        root_rule = next(iter(p_grammar.grammar.rules[p_grammar.grammar.root_nt]))
        self.root_element = InsideElement(Score(p_grammar.parameter[root_rule.uid]))
        self.p_grammar = p_grammar

    def initialize_element_from_rule(self, rule: GrammarRule) -> InsideElement:
        assert self.p_grammar is not None
        return InsideElement(Score(self.p_grammar.parameter[rule.uid]))

########################## PARALLELIZED PROCESSING ##########################

def logspace_matmul(mat1, mat2):
    raise NotImplementedError

def inside_matmul(mat1, mat2, matmul_op=lambda m1,m2: m1 @ m2):
    assert mat1.shape[-1] == mat2.shape[0]
    assert len(mat1.shape) in {
        1,
        2,
    }, "matmul of higher-dimensional (>2) matrices not supported yet."
    assert len(mat2.shape) in {
        1,
        2,
    }, "matmul of higher-dimensional (>2) matrices not supported yet."

    if not (mat1.is_sparse) and not (mat2.is_sparse):
        return matmul_op(mat1, mat2)
    elif mat1.is_sparse:
        return matmul_op(mat1, mat2)

    if len(mat1.shape) == 1:
        if len(mat2.shape) == 1:
            return matmul_op(mat2, mat1)

        return matmul_op(mat2.transpose(1, 0), mat1)

    if len(mat2.shape) == 1:
        return matmul_op(mat2, mat1.transpose(1, 0))

    return matmul_op(mat2.transpose(1, 0), mat1.transpose(1, 0)).transpose(1, 0)
