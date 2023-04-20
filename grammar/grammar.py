from __future__ import annotations
import json
import ast
import itertools
import copy
import random
import numpy as np
import torch

import typing
from typing import NewType, List, Dict, Tuple, Set, Optional, Iterable, Union, Any
from collections.abc import Sequence, MutableSequence, Mapping, MutableMapping

Symbol = NewType("Symbol", int)
NonTerminal = NewType("NonTerminal", Symbol)
Terminal = NewType("Terminal", Symbol)

RuleId = NewType("RuleId", int)
Score = NewType("Score", float)

NonTerminalIndex = NewType("NonTerminalIndex", int)  # count-wise
RuleSeqIndex = NewType("RuleSeqIndex", int)  # token-wise after tokenization

class Derivation:
    def __init__(
        self,
        root_rule: GrammarRule,
        children: Mapping[Tuple[NonTerminal, NonTerminalIndex], Derivation | None],
    ):
        """Derivations are a GrammarRule and its children represented as a dictionary mapping
        indexed nonterminal introduced by GrammarRule to list of Derivations describing it.
        """
        self.root_rule = root_rule
        self.children = children

    @staticmethod
    def create_derivation_with_empty_children(rule: GrammarRule) -> Derivation:
        empty_children: Mapping[
            Tuple[NonTerminal, NonTerminalIndex], Derivation | None
        ] = {
            (NonTerminal(rule.src[idx_in_src]), NonTerminalIndex(child_nt_idx)): None
            for child_nt_idx, (idx_in_src, _) in rule.corresp.items()
        }
        return Derivation(rule, empty_children)

    def __str__(self):
        def format_string(derivation: Derivation | None, tab_amt=0):
            if derivation is None:
                return "\t" * tab_amt + "None"
            s = "\t" * tab_amt + str(derivation.root_rule)
            for nt, d in derivation.children.items():
                s += "\n" + "\t" * tab_amt + f"->{nt[0]}_{nt[1]}:"
                s += "\n" + format_string(d, tab_amt + 1)
            return s

        return format_string(self)

    def rules(self) -> List[GrammarRule]:
        rules_in_deriv = [self.root_rule]
        for _, child_deriv in self.children.items():
            if child_deriv is None:
                continue
            rules_in_deriv.extend(child_deriv.rules())
        return rules_in_deriv

    def get_produced_src_string(self, tokenizer) -> str:
        if len(self.children) == 0:
            return tokenizer.decode(self.root_rule.src)

        decoded_src_str = tokenizer.decode(self.root_rule.src)
        idx_of_nt_in_src = len(decoded_src_str)
        for ((child_nt, _), deriv) in reversed(self.children.items()):
            child_nt_str = tokenizer.decode(child_nt)
            idx_of_nt_in_src = decoded_src_str.rfind(child_nt_str, 0, idx_of_nt_in_src)
            if deriv:
                subderiv_string = deriv.get_produced_src_string(tokenizer)
                decoded_src_str = (
                    decoded_src_str[:idx_of_nt_in_src]
                    + subderiv_string
                    + decoded_src_str[idx_of_nt_in_src + len(child_nt_str) :]
                )
            else:
                decoded_src_str = (
                    decoded_src_str[:idx_of_nt_in_src]
                    + "UNEXPANDED"
                    + decoded_src_str[idx_of_nt_in_src + len(child_nt_str) :]
                )
        return decoded_src_str

    def update_child_derivation(
        self,
        nt_tok: NonTerminal,  # not indexed
        nt_idx: NonTerminalIndex,
        child_derivation: Derivation,
    ) -> Derivation:
        assert (nt_tok, nt_idx) in self.children
        new_children: Mapping[
            Tuple[NonTerminal, NonTerminalIndex], Derivation | None
        ] = {
            k: copy.deepcopy(v) if k != (nt_tok, nt_idx) else child_derivation
            for k, v in self.children.items()
        }
        return Derivation(self.root_rule, new_children)

    @property
    def is_complete(self) -> bool:
        if len(self.children) == 0:
            return True
        if any(child_deriv is None for child_deriv in self.children.values()):
            return False
        for child_deriv in self.children.values():
            assert child_deriv is not None
            if not child_deriv.is_complete:
                return False
        return True

    def expand_leftmost(
        self, p_grammar: ProbabilisticGrammar
    ) -> Iterable[Tuple[Derivation, Score]]:
        for (child_nt, child_nt_idx), child_deriv in self.children.items():
            if child_deriv is None:
                for nt_rule in p_grammar.grammar.rules[child_nt]:
                    new_child_deriv = Derivation.create_derivation_with_empty_children(
                        nt_rule
                    )
                    yield self.update_child_derivation(
                        child_nt, child_nt_idx, new_child_deriv
                    ), Score(p_grammar.parameter[nt_rule.uid])
                return
            assert child_deriv is not None
            if child_deriv.is_complete:
                continue
            for new_child_deriv, expansion_score in child_deriv.expand_leftmost(
                p_grammar
            ):
                yield self.update_child_derivation(
                    child_nt, child_nt_idx, new_child_deriv
                ), expansion_score
            return

    def concatenate(self, child: Derivation) -> Derivation:
        # Concatenates to first (leftmost) open child with the correct nonterminal.
        if self.is_complete:
            print(f"{self} is a complete derivation. Cannot concatenate to it.")
            return self
        for (child_nt, child_nt_idx), child_deriv in self.children.items():
            if child_deriv is None and child_nt == child.root_rule.nt:
                return self.update_child_derivation(child_nt, child_nt_idx, child)
            if child_deriv is None or child_deriv.is_complete:
                continue
            assert child_deriv is not None
            return self.update_child_derivation(
                child_nt, child_nt_idx, child_deriv.concatenate(child)
            )
        assert False, f"Could not concatenate {child} onto {self}"

    def equals(self, other: Derivation) -> bool:
        if self.root_rule != other.root_rule:
            return False
        if len(self.children) != len(other.children):
            return False
        for (self_k, self_v) in self.children.items():
            if self_k not in other.children:
                return False
            other_child = other.children[self_k]
            if self_v is None:
                if other_child is not None:
                    return False
                continue
            if other_child is None:
                return False
            assert self_v is not None
            assert other_child is not None
            if not self_v.equals(other_child):
                return False
        return True


import heapq


class MaxHeap:
    def __init__(self, in_logspace=True):
        self.heap = []
        self.counter = 0  # to break ties

    def push(self, prob, item):
        self.counter += 1
        heapq.heappush(self.heap, (-prob, self.counter, item))
        # the smaller the prob, lesser priority
        # the more negative the logprob (smaller probability), lesser priority also

    def pop(self):
        assert not self.empty()
        neg_prob, _, item = heapq.heappop(self.heap)
        return -neg_prob, item

    def empty(self):
        return len(self.heap) == 0


class GrammarRule:
    def __init__(
        self,
        nt: NonTerminal,
        src: Sequence[Symbol],
        tgt: Sequence[Symbol],
        corresp: Mapping[NonTerminalIndex, Tuple[RuleSeqIndex, RuleSeqIndex]],
        unique_id: RuleId,
        star: Sequence[Tuple[RuleSeqIndex, RuleSeqIndex]] = [],
    ):
        self.nt = nt
        self.src = src
        self.tgt = tgt
        self.corresp = corresp
        self.uid = unique_id
        self.star = star

    def get_as_str(self, tokenizer):
        nt_as_str = tokenizer.decode(self.nt)
        src_strs = tokenizer.convert_ids_to_tokens(self.src)
        tgt_strs = tokenizer.convert_ids_to_tokens(self.tgt)
        for nt_i, (src_i, tgt_i) in self.corresp.items():
            assert (
                self.src[src_i] == self.tgt[tgt_i]
            ), f"Inconsistent between {src_as_str} and {tgt_as_str} at tokenized indices {src_i} and {tgt_i}"
            src_strs[src_i] += f"_{nt_i}"
            tgt_strs[tgt_i] += f"_{nt_i}"

        return f"{nt_as_str}->\n\t\t{' '.join(src_strs)}\n\t\t{' '.join(tgt_strs)}"

    def __str__(self):
        src_indexed = [str(t) for t in self.src]
        tgt_indexed = [str(t) for t in self.tgt]
        for i, (src_i, tgt_i) in reversed(self.corresp.items()):
            assert (
                src_indexed[src_i] == tgt_indexed[tgt_i]
            ), f"Inconsistent between {src_indexed} and {tgt_indexed} at indices {src_i} and {tgt_i}"
            nt = f"{self.src[src_i]}_{i}"
            src_indexed[src_i] = nt
            tgt_indexed[tgt_i] = nt
        src_indexed = " ".join(src_indexed)
        tgt_indexed = " ".join(tgt_indexed)
        return f"({self.uid}) {self.nt}->\t{src_indexed} ; \t{tgt_indexed}"

    def __eq__(self, other):
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)


class Grammar:
    def __init__(
        self,
        nonterminals: Set[NonTerminal],
        src_terminals: Set[Terminal],
        tgt_terminals: Set[Terminal],
        rules: MutableMapping[NonTerminal, Set[GrammarRule]],  # unindexed NT ->
        root: NonTerminal,
    ):
        self.nonterminals = nonterminals
        self.src_terminals = src_terminals
        self.tgt_terminals = tgt_terminals
        self.rules = rules
        self.root_nt = root

    @staticmethod
    def from_files(rules_filepath: str, grammar_config_filepath: str) -> Grammar:
        rules: MutableMapping[NonTerminal, Set[GrammarRule]] = {}
        num_rules = 0
        nonterminals: Set[NonTerminal] = set()
        src_terminals: Set[Terminal] = set()
        tgt_terminals: Set[Terminal] = set()

        with open(grammar_config_filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

            nonterminals = {NonTerminal(nt) for nt in config_dict["nonterminals"]}
            src_terminals = {Terminal(t) for t in config_dict["src_terminals"]}
            tgt_terminals = {Terminal(t) for t in config_dict["tgt_terminals"]}
            root_nt = config_dict["start_symbol"]
            assert type(root_nt) == int
            assert root_nt in nonterminals

        with open(rules_filepath, "r", encoding="utf-8") as f:
            translation_dict = json.load(f)
            for nt, nt_rules in translation_dict.items():
                nt = int(nt)
                nt_typed = NonTerminal(nt)
                if nt_typed not in rules:
                    rules[nt_typed] = set()
                for rule in nt_rules:
                    typed_src: List[Symbol] = []
                    for s in rule["src"]:
                        if s in nonterminals:
                            typed_src.append(NonTerminal(s))
                        else:  # if s in src_terminals:
                            typed_src.append(Terminal(s))
                        # else:
                        #     assert (
                        #         False
                        #     ), f"Symbol {s} not recognized as a terminal or nonterminal"

                    typed_tgt: List[Symbol] = []
                    for s in rule["tgt"]:
                        if s in nonterminals:
                            typed_tgt.append(NonTerminal(s))
                        else:  # if s in tgt_terminals:
                            typed_tgt.append(Terminal(s))
                        # else:
                        #     assert (
                        #         False
                        #     ), f"Symbol {s} not recognized as a terminal or nonterminal"

                    corresp = rule["align"]
                    rules[nt].add(
                        GrammarRule(
                            nt_typed,
                            typed_src,
                            typed_tgt,
                            corresp,
                            RuleId(num_rules),
                            rule["star"],
                        )
                    )
                    num_rules += 1

        return Grammar(
            nonterminals,
            src_terminals,
            tgt_terminals,
            rules,
            NonTerminal(Symbol(root_nt)),
        )

    def write_to_file(self, rules_filepath: str, config_filepath: str) -> None:
        jsonable_grammar: MutableMapping[
            int,
            MutableSequence[
                Mapping[
                    str,
                    Union[
                        Sequence[int],
                        Mapping[int, Tuple[int, int]],
                        Sequence[Tuple[int, int]],
                    ],
                ]
            ],
        ] = {}
        for nt, rules in self.rules.items():
            if nt not in jsonable_grammar:
                jsonable_grammar[nt] = []
            for rule in rules:
                assert nt == rule.nt
                jsonable_grammar[rule.nt].append(
                    {
                        "src": [int(t) for t in rule.src],
                        "tgt": [int(t) for t in rule.tgt],
                        "align": {
                            int(k): (int(v[0]), int(v[1]))
                            for k, v in rule.corresp.items()
                        },
                        "star": rule.star,  # [[s, e] for (s, e) in rule.star],
                    }
                )

        with open(rules_filepath, "w", encoding="utf-8") as f:
            json.dump(jsonable_grammar, f, indent=2)
        with open(config_filepath, "w", encoding="utf-8") as f:
            config = {
                "start_symbol": self.root_nt,
                "nonterminals": list(self.nonterminals),
                "src_terminals": list(self.src_terminals),
                "tgt_terminals": list(self.tgt_terminals),
            }
            json.dump(config, f, indent=2)

    def write_to_file_as_str(self, rules_filepath: str, tokenizer) -> None:
        jsonable_grammar: MutableMapping[
            str,
            MutableSequence[
                Mapping[
                    str,
                    Union[
                        Sequence[str],
                        Mapping[int, Tuple[int, int]],
                        Sequence[Tuple[int, int]],
                    ],
                ]
            ],
        ] = {}
        for rule_nt, rules in self.rules.items():
            nt_str = tokenizer.decode(rule_nt)
            if nt_str not in jsonable_grammar:
                jsonable_grammar[nt_str] = []
            for rule in rules:
                src_str = tokenizer.decode(rule.src)
                tgt_str = tokenizer.decode(rule.tgt)
                jsonable_grammar[nt_str].append(
                    {
                        "src": src_str,
                        "tgt": tgt_str,
                        "align": {
                            int(k): (int(v[0]), int(v[1]))
                            for k, v in rule.corresp.items()
                        },
                        "star": rule.star,
                    }
                )

        with open(rules_filepath, "w", encoding="utf-8") as f:
            json.dump(jsonable_grammar, f, indent=2)

    def derivation_generator(
        self, nt: NonTerminal, max_depth: int = -1
    ) -> Iterable[Derivation]:

        for rule in self.rules[nt]:
            if len(rule.corresp) == 0:
                yield Derivation(rule, {})
                continue
            if max_depth == 0:
                continue

            children_options: MutableMapping[
                Tuple[NonTerminal, NonTerminalIndex], Sequence[Derivation]
            ] = {
                (NonTerminal(rule.src[src_idx]), nt_idx): []
                for (nt_idx, (src_idx, _)) in rule.corresp.items()
            }
            cannot_expand = False
            for (child_nt_idx, (src_idx, tgt_idx)) in rule.corresp.items():
                child_nt = NonTerminal(rule.src[src_idx])
                child_deriv_options = list(
                    self.derivation_generator(child_nt, max_depth - 1)
                )
                if len(child_deriv_options) == 0:
                    cannot_expand = True
                    break
                children_options[(child_nt, child_nt_idx)] = child_deriv_options
            if cannot_expand:
                continue

            for deriv_combo in itertools.product(*children_options.values()):
                children: MutableMapping[
                    Tuple[NonTerminal, NonTerminalIndex], Derivation | None
                ] = {k: None for k in children_options.keys()}
                for (subnt, subderiv) in zip(children_options.keys(), deriv_combo):
                    children[subnt] = subderiv
                yield Derivation(rule, children)

    def add_root_rule(self, new_root_nt: NonTerminal) -> RuleId:
        assert (
            new_root_nt not in self.rules
        ), f"Nonterminal {new_root_nt} already in grammar."
        new_rule_id = RuleId(sum(len(rules) for rules in self.rules.values()))
        new_root_rule = GrammarRule(
            NonTerminal(new_root_nt),
            [self.root_nt],
            [self.root_nt],
            {NonTerminalIndex(0): (RuleSeqIndex(0), RuleSeqIndex(0))},
            new_rule_id,
            [],
        )
        self.nonterminals.add(new_root_nt)
        self.rules[new_root_nt] = {new_root_rule}
        self.root_nt = new_root_nt
        return new_rule_id

    def num_rules(self) -> int:
        return sum(len(rules) for rules in self.rules.values())


def create_grammar_rule(
    nt: NonTerminal,
    abs_code: Sequence[Symbol],
    abs_text: Sequence[Symbol],
    correspondences: Mapping[
        NonTerminalIndex, Tuple[RuleSeqIndex, RuleSeqIndex]
    ],  # Note that we actually cant have multiple index on the same side...
    num_rules: RuleId,
    star: Sequence[Tuple[RuleSeqIndex, RuleSeqIndex]] = [],
) -> GrammarRule:
    abs_code_typed: Sequence[Symbol] = [Symbol(s) for s in abs_code]
    abs_text_typed: Sequence[Symbol] = [Symbol(s) for s in abs_text]
    correspondences_typed = {
        NonTerminalIndex(k): (v[0], v[1]) for k, v in correspondences.items()
    }
    return GrammarRule(
        NonTerminal(Symbol(nt)),
        abs_code_typed,
        abs_text_typed,
        correspondences_typed,
        RuleId(num_rules),
        star,
    )


def create_grammar(
    nonterminals: Set[NonTerminal],
    src_terminals: Set[Terminal],
    tgt_terminals: Set[Terminal],
    rules: Mapping[NonTerminal, Set[GrammarRule]],
    root: NonTerminal,
) -> Grammar:
    typed_nonterminals: Set[NonTerminal] = {
        NonTerminal(Symbol(s)) for s in nonterminals
    }
    typed_src_terminals: Set[Terminal] = {Terminal(Symbol(s)) for s in src_terminals}
    typed_tgt_terminals: Set[Terminal] = {Terminal(Symbol(s)) for s in tgt_terminals}
    typed_rules: MutableMapping[NonTerminal, Set[GrammarRule]] = {
        NonTerminal(Symbol(k)): v for k, v in rules.items()
    }
    typed_root: NonTerminal = NonTerminal(Symbol(root))

    return Grammar(
        typed_nonterminals,
        typed_src_terminals,
        typed_tgt_terminals,
        typed_rules,
        typed_root,
    )


class ProbabilisticGrammar:
    def __init__(self, grammar: Grammar, parameter: Mapping[RuleId, float]):
        ProbabilisticGrammar.assert_parameter_matches_grammar(grammar, parameter)
        self.grammar = grammar
        self.parameter = parameter

    @staticmethod
    def assert_parameter_matches_grammar(
        grammar: Grammar, parameter: Mapping[RuleId, float]
    ):
        assert len(parameter) == sum(
            len(nt_rules) for nt_rules in grammar.rules.values()
        ), f"{len(parameter)} rules in parameter, but {sum(len(nt_rules) for nt_rules in grammar.rules.values())} rules in grammar."
        all_rule_ids = {r.uid for rules in grammar.rules.values() for r in rules}
        assert set(parameter.keys()) == all_rule_ids

    @staticmethod
    def create_random_parameter(
        grammar: Grammar, in_logspace=True
    ) -> ProbabilisticGrammar:
        parameter: MutableMapping[RuleId, float] = {}
        for nt, rules in grammar.rules.items():
            nt_parameters = {r.uid: np.exp(random.uniform(0, 1)) for r in rules}
            sumexp = sum(nt_parameters.values())
            for r_uid in nt_parameters:
                if in_logspace:
                    parameter[r_uid] = np.log(nt_parameters[r_uid] / sumexp)
                else:
                    parameter[r_uid] = nt_parameters[r_uid] / sumexp

        return ProbabilisticGrammar(grammar, parameter)

    def top_down_generator(
        self, max_num_generated=-1, in_logspace=True
    ) -> Iterable[Tuple[Derivation, Score]]:
        """Generate programs in order of decreasing probability."""

        heap = MaxHeap()
        for root_rule in self.grammar.rules[self.grammar.root_nt]:
            heap.push(
                Score(self.parameter[root_rule.uid]),
                Derivation.create_derivation_with_empty_children(root_rule),
            )

        while not heap.empty():
            prob, partial_deriv = heap.pop()
            if partial_deriv.is_complete:
                yield partial_deriv, prob
                max_num_generated -= 1
                if max_num_generated == 0:
                    break
            else:
                for (
                    new_partial_deriv,
                    incremental_prob,
                ) in partial_deriv.expand_leftmost(self):
                    if in_logspace:
                        heap.push(prob + incremental_prob, new_partial_deriv)
                    else:
                        heap.push(prob * incremental_prob, new_partial_deriv)

    def __str__(self):
        self_string = ""
        for nt, rules in self.grammar.rules.items():
            for rule in rules:
                weight = self.parameter[rule.uid]
                self_string += f"{str(rule)} ({weight})\n"
        return self_string
