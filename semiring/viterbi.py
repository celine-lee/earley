from __future__ import annotations

from dataclasses import dataclass
from typing import Set

import numpy as np
import torch

from semiring import Element, Semiring
from grammar.grammar import (
    Score,
    Derivation,
    NonTerminal,
    ProbabilisticGrammar,
    GrammarRule,
    Symbol,
)

# from grammar.derivation import Derivation

""" Viberbi """

########################## SERIAL PROCESSING ##########################



@dataclass
class ViterbiElement(Element):
    """A Viterbi element is a (score,  Derivations with that score) representation in semiring parsing."""

    score: torch.Tensor  # R_0^1
    derivations: Set[Derivation]  # 2^E

    def mul(self, other: ViterbiElement) -> ViterbiElement:
        # multiply scores (add logprobs), concat deriv forests; not nec commutative
        new_score = self.score + other.score
        new_derivations: Set[Derivation] = set()
        if self.derivations == {None}:
            new_derivations = other.derivations
        elif other.derivations == {None}:
            new_derivations = self.derivations
        else:
            for parent_deriv in self.derivations:
                assert parent_deriv is not None
                for child_deriv in other.derivations:
                    assert child_deriv is not None
                    new_deriv = parent_deriv.concatenate(child_deriv)
                    new_derivations.add(new_deriv)
        return ViterbiElement(new_score, new_derivations)

    def add(self, other: ViterbiElement) -> ViterbiElement:
        # max score, keep both in case of ties; commutative
        if self.score > other.score:
            return self
        if other.score > self.score:
            return other
        combined_derivations = self.derivations | other.derivations
        if len(combined_derivations) > 1 and None in combined_derivations:
            combined_derivations.remove(None)
        return ViterbiElement(self.score, combined_derivations)

    def __eq__(self, other) -> bool:
        if not torch.isclose(self.score, other.score):
            return False
        if len(self.derivations) != len(other.derivations):
            return False
        if self.derivations == {None}:
            return other.derivations == {None}
        if other.derivations == {None}:
            return False
        for d in self.derivations:
            assert d is not None
            exists_in_other = False
            for o_d in other.derivations:
                assert o_d is not None
                if d.equals(o_d):
                    exists_in_other = True
                    break

            if not exists_in_other:
                return False
        return True

    def __str__(self):
        return (
            f"(Score: {self.score})"
            + "\n"
            + "\n".join([str(d) for d in self.derivations])
        )

    def get_score(self):
        return self.score, self.derivations


class ViterbiSemiring(Semiring):
    def __init__(self, p_grammar: ProbabilisticGrammar):
        super().__init__(
            mul_id=ViterbiElement(torch.log(torch.tensor([1.0])), {None}),
            add_id=ViterbiElement(torch.tensor([-100], dtype=torch.float64), set()),
            # add_id=ViterbiElement(torch.log(torch.tensor([1e-10])), set()),
        )
        root_rule = next(iter(p_grammar.grammar.rules[p_grammar.grammar.root_nt]))
        self.root_element = ViterbiElement(
            p_grammar.parameter[root_rule.uid],
            {Derivation.create_derivation_with_empty_children(root_rule)},
        )
        self.p_grammar = p_grammar

    def initialize_element_from_rule(self, rule: GrammarRule) -> ViterbiElement:
        assert self.p_grammar is not None
        return ViterbiElement(
            self.p_grammar.parameter[rule.uid],
            {Derivation.create_derivation_with_empty_children(rule)},
        )


########################## PARALLELIZED PROCESSING ##########################


def viterbi_matmul(mat1, mat2):
    if mat1.shape[-1] != mat2.shape[0]: assert mat1.shape[-1] == mat2.shape[1]
    else: assert mat1.shape[-1] == mat2.shape[0]

    m1 = mat1
    if mat1.is_sparse:
        m1 = mat1.to_dense()
    m2 = mat2
    if mat2.is_sparse:
        m2 = mat2.to_dense()

    if len(m1.shape) == 1:
        if len(m2.shape) == 1:
            return torch.max(m1 * m2)
        return torch.max(m1 * m2, 0)[0]
    if len(m2.shape) == 1:
        return torch.max(m1 * m2, 1)[0]
    if len(m1.shape) == len(m2.shape):
        if len(m1.shape) == 2: return torch.max(m1[..., None] * m2[None, ...], 1)[0]
        else: return torch.max(m1.reshape(mat1.shape[0],mat1.shape[1],mat1.shape[2],1) * m2.reshape(mat2.shape[0],1,mat2.shape[1],mat2.shape[2]), 2)[0]

    if len(m2.shape) == 2:
        assert len(m1.shape) == 3
        return torch.max(m1.reshape(*mat1.shape,1) * m2.reshape(1,1,*mat2.shape), 2)[0]
    if len(m1.shape) == 2:
        assert len(m2.shape) == 3
        return torch.max(m1.reshape(*mat1.shape,1,1) * m2.reshape(1,*mat2.shape), 2)[0]



