from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, NewType
import torch
import itertools

from grammar.grammar import (
    Symbol,
    Terminal,
    GrammarRule,
    ProbabilisticGrammar,
    Grammar,
    RuleId,
)
from grammar.tokenizer import Tokenizer

from earley_parsing.earley import DotPosition

DotStateId = NewType("DotStateId", int)


def pretty_print_dp_scores(grammar, dp_scores, tokenizer):
    # For debugging; print DP scores in readable way.
    rule_id_to_grammarrule = {}
    for rules in grammar.rules.values():
        for r in rules:
            assert r.uid not in rule_id_to_grammarrule
            rule_id_to_grammarrule[r.uid] = r

    dotstate_vocabulary_reversed = {
        v: (rule_id_to_grammarrule[k[0]], k[1]) for k, v in V.items()
    }

    N, _, V = dp_scores.shape
    for end_pos in range(N):
        print("j=", end_pos)
        for origin in range(end_pos + 1):
            print("\ti=", origin)
            for dotstate_id in range(V):
                rule = dotstate_vocabulary_reversed[dotstate_id][0]
                dot_pos = dotstate_vocabulary_reversed[dotstate_id][1]
                before_dot_seq = " ".join(
                    [tokenizer.decode(t) for t in rule.src[:dot_pos]]
                )
                after_dot_seq = " ".join(
                    [tokenizer.decode(t) for t in rule.src[dot_pos:]]
                )
                dotted_rule = f"{tokenizer.decode(rule.nt)} -> {before_dot_seq} <DOT> {after_dot_seq}"
                print(
                    "\t\t", dotted_rule, " : ", dp_scores[end_pos, origin, dotstate_id]
                )


@dataclass
class EarleySupport:
    """
    Support matrices and dictionary for parallelized Earley parsing.

    V: dotstate vocabulary. Maps dotstate, a pair of (grammar rule, dot position), to unique dotstate ID
    root_rule: GrammarRule that corresponds to the root ROOT->START rule of the grammar
    P: grammar rule probability for Earley prediction. R^|V| : dotstate to grammar rule probability if dot position is 0, to semiring 0 otherwise.
    E: expectation matrix. {0,1}^|V|x|V| (sparse) : maps dotstates to the dotstates it expects in Earley completion. 1 if the second index maps to a complete dotstate whose left-side nonterminal is the next symbol in the first index dotstate
            e.g. E[A->alpha DOT B beta, B->gamma DOT] = 1; E[A->alpha DOT B beta, B->DOT gamma] = 0; E[A->alpha DOT B beta, C->gamma DOT] = 0
    T: transition matrix. {0,1}^|S|x|V|x|V| (sparse) : maps dotstate to the dotstate that is advanced one dot index to pass the symbol of the first dimension, if it exists
            e.g. T[B, A->alpha DOT B beta, A->alpha B DOT beta] = 1; T[beta, A->alpha DOT B beta, A->alpha B DOT beta] = 0; T[B, A->alpha DOT B beta, B->gamma DOT] = 0
    all_T: all transisions matrix. {0,1}^|V|x|V| (sparse) : maps dotstate to the dotstate that is advanced one dot index. equal to T.sum(0)
            e.g. T[A->alpha DOT B beta, A->alpha B DOT beta] = 1; T[A->alpha DOT B beta, B->gamma DOT] = 0
    """

    V: Dict[Tuple[RuleId, DotPosition], DotStateId]
    root_rule: GrammarRule
    P: torch.Tensor
    E: torch.Tensor
    T: torch.Tensor
    all_T: torch.Tensor

    @staticmethod
    def initialize_parallelization_matrices(
        p_grammar: ProbabilisticGrammar, tokenizer: Tokenizer, semiring_zero: float
    ):
        grammar = p_grammar.grammar

        assert (
            len(grammar.rules[grammar.root_nt]) == 1
        ), "There should be one root rule."
        root_rule = next(iter(grammar.rules[grammar.root_nt]))
        assert len(root_rule.src) == 1, "Root rule should just be length 1."

        dotstate_vocabulary: Dict[Tuple[RuleId, DotPosition], DotStateId] = {}

        for nt, rules in grammar.rules.items():
            for rule in rules:
                for dot_idx in range(len(rule.src) + 1):
                    dotstate_id = DotStateId(len(dotstate_vocabulary))
                    dotstate_vocabulary[(rule.uid, DotPosition(dot_idx))] = dotstate_id

        num_dotstate_pairs = len(dotstate_vocabulary)
        total_num_symbols = len(tokenizer)

        dotstate_prediction_scores = torch.ones((num_dotstate_pairs)) * semiring_zero
        expectation_matrix = torch.zeros((num_dotstate_pairs, num_dotstate_pairs))
        transition_matrix = torch.zeros(
            (total_num_symbols, num_dotstate_pairs, num_dotstate_pairs)
        )

        for rule in itertools.chain.from_iterable(grammar.rules.values()):
            for dot_idx in range(len(rule.src) + 1):
                dotstate_id = dotstate_vocabulary[(rule.uid, DotPosition(dot_idx))]
                if dot_idx == 0:
                    dotstate_prediction_scores[dotstate_id] = p_grammar.parameter[
                        rule.uid
                    ]
                if dot_idx < len(rule.src):
                    next_symbol = rule.src[dot_idx]
                    next_dotstate_id = dotstate_vocabulary[
                        (rule.uid, DotPosition(dot_idx + 1))
                    ]
                    transition_matrix[next_symbol][dotstate_id][next_dotstate_id] = 1

                if next_symbol in grammar.nonterminals:
                    for child_rule in grammar.rules[next_symbol]:
                        child_dotstate_id = dotstate_vocabulary[
                            (child_rule.uid, DotPosition(len(child_rule.src)))
                        ]
                        expectation_matrix[dotstate_id, child_dotstate_id] = 1

        return EarleySupport(
            dotstate_vocabulary,
            root_rule,
            dotstate_prediction_scores,
            expectation_matrix.to_sparse(),
            transition_matrix.to_sparse(),
            transition_matrix.sum(0).to_sparse(),
        )


def earley_semiring_parallelize(
    input: List[Terminal],
    support: EarleySupport,
    add: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    matmul: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    mul: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    add_id: float,
):
    input.append(Terminal(Symbol(-1)))
    N = len(input)

    # R^NxNx|V|; entry [end_pos, origin, dotstate] is the score of Earley item [origin, dotstate, end_pos]
    dp_scores = torch.ones((N, N, len(support.V))) * add_id

    # PREDICT. Since all items with the same origin and end_pos are generated from prediction steps, do all predictions in advance.
    for index in range(N):
        dp_scores[index, index] = support.P

    # Process items according to the partial order. An item is processed either by being COMPLETEd or SCANned.
    for end_pos in range(N):
        for split in reversed(range(end_pos)):
            # COMPLETE. For any complete item over range `[split, end_pos]`, `mul` its score with any parent item with end index `split` and `add` result to new joint item.
            # R^Nx|V| score of every parent item [origin (free), dotstate, split]
            parent_scores = dp_scores[split]

            # R^|V| combined score of all complete children in `[split, end_pos]`, combined across and indexed by shared parent item
            scores_by_parent = matmul(support.E, dp_scores[end_pos, split])
            # R^|N|x|V| combined score of parent item and all of its complete children, resulting in new joint item. indexed by parent item
            parent_all_children_score = mul(parent_scores, scores_by_parent[None, :])
            # R^|N|x|V| score of new joint item, indexed by joint item
            joint_item_score = matmul(parent_all_children_score, support.all_T)

            # `add` newly processed joint scores to existing scores.
            dp_scores[end_pos] = add(dp_scores[end_pos], joint_item_score)

        # SCAN. For items whose next symbol is the current symbol, copy scores into next time step with dot advanced (scanned).
        if end_pos < N - 1:
            new_scan_scores = matmul(dp_scores[end_pos], support.T[input[end_pos]])
            dp_scores[end_pos + 1] = add(dp_scores[end_pos + 1], new_scan_scores)

    # Return the score of the completed root rule.
    return dp_scores[N - 1, 0, support.V[(support.root_rule.uid, DotPosition(1))]]
