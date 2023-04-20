from __future__ import annotations
from dataclasses import dataclass
from typing import NewType, List, Dict, Iterable, Set
from collections.abc import MutableMapping

from semiring.semiring import Item, Semiring, Element, Chart
from grammar.grammar import (
    Symbol,
    Terminal,
    NonTerminal,
    GrammarRule,
    ProbabilisticGrammar,
    Grammar,
)
from semiring.boolean import BooleanSemiring
from semiring.viterbi import ViterbiSemiring
from semiring.inside import InsideSemiring


SentencePosition = NewType("SentencePosition", int)
DotPosition = NewType("DotPosition", int)


@dataclass
class EarleyItem(Item):
    origin_pos: SentencePosition  # i
    rule: GrammarRule
    dot_idx: DotPosition
    end_pos: SentencePosition  # j

    def __hash__(self):
        return hash((self.origin_pos, self.rule.uid, self.dot_idx, self.end_pos))

    @property
    def is_done(self):
        return self.dot_idx == len(self.rule.src)

    def __str__(self):
        before_dot_seq = " ".join([str(t) for t in self.rule.src[: self.dot_idx]])
        after_dot_seq = " ".join([str(t) for t in self.rule.src[self.dot_idx :]])
        dotted_rule = f"{self.rule.nt} -> {before_dot_seq} <DOT> {after_dot_seq}"
        return f"[{self.origin_pos}, {dotted_rule}, {self.end_pos}]"

    def predict(
        self,
        relevant_grammar_rules: MutableMapping[NonTerminal, Set[GrammarRule]],
        curr_pos: SentencePosition,
    ) -> Iterable[EarleyItem]:
        next_nt = self.rule.src[self.dot_idx]  # B
        for r in relevant_grammar_rules[NonTerminal(next_nt)]:  # (B -> gamma)
            yield EarleyItem(curr_pos, r, DotPosition(0), curr_pos)

    def scan(self, curr_pos: SentencePosition) -> EarleyItem:
        return EarleyItem(
            self.origin_pos,
            self.rule,
            DotPosition(self.dot_idx + 1),
            SentencePosition(curr_pos + 1),
        )


class EarleyChart(Chart):
    def __init__(self, semiring: Semiring, goal_state: EarleyItem):
        self.semiring = semiring
        self.goal_state = goal_state
        self.chart: Dict[EarleyItem, Element] = {}  # item -> semiring element
        self.chart_by_end_pos: Dict[
            SentencePosition, Dict[EarleyItem, Element]
        ] = {}  # by j
        self.chart_by_origin_pos: Dict[
            SentencePosition, Dict[EarleyItem, Element]
        ] = {}  # by i

    def __getitem__(self, item: EarleyItem) -> Element:
        if item in self.chart:
            return self.chart[item]
        return self.semiring.add_id

    def add_element(self, item: EarleyItem, element: Element) -> Element:
        self.chart[item] = self[item].add(element)
        element_at_item = self.chart[item]
        if item.end_pos not in self.chart_by_end_pos:
            self.chart_by_end_pos[item.end_pos] = {item: element_at_item}
        else:
            self.chart_by_end_pos[item.end_pos][item] = element_at_item
        if item.origin_pos not in self.chart_by_origin_pos:
            self.chart_by_origin_pos[item.origin_pos] = {item: element_at_item}
        else:
            self.chart_by_origin_pos[item.origin_pos][item] = element_at_item
        return element_at_item

    def get_chart_by_end_pos(
        self, end_pos: SentencePosition
    ) -> Dict[EarleyItem, Element]:
        if end_pos not in self.chart_by_end_pos:
            return {}
        return self.chart_by_end_pos[end_pos]

    def get_chart_by_origin_pos(
        self, origin_pos: SentencePosition
    ) -> Dict[EarleyItem, Element]:
        if origin_pos not in self.chart_by_origin_pos:
            return {}
        return self.chart_by_origin_pos[origin_pos]

    def get_goal_state(self) -> Element:
        return self[self.goal_state]


def filter_grammar(
    all_grammar_rules: MutableMapping[NonTerminal, Set[GrammarRule]],
    input: List[Terminal],
):
    filtered_grammar: MutableMapping[NonTerminal, Set[GrammarRule]] = {}
    input_terminals = set(input)
    nonterminals = set(all_grammar_rules.keys())
    for nt, rules in all_grammar_rules.items():
        filtered_grammar[nt] = set()
        for rule in rules:
            rule_terminals = set(rule.src) - nonterminals
            if (len(rule_terminals) == 0) or (
                len(rule_terminals & input_terminals) > 0
            ):
                filtered_grammar[nt].add(rule)
        if len(filtered_grammar[nt]) == 0:
            del filtered_grammar[nt]
    return filtered_grammar


def earley_semiring(semiring: Semiring, input: List[Terminal]) -> Element:
    input.append(Terminal(Symbol(-1)))
    N = len(input)

    p_grammar = semiring.p_grammar
    assert p_grammar is not None
    grammar = p_grammar.grammar

    assert len(grammar.rules[grammar.root_nt]) == 1, "There is more than one root rule."
    root_rule = next(iter(grammar.rules[grammar.root_nt]))
    assert len(root_rule.corresp) == 1, "Root rule should be unary."

    potentially_relevant_rules: MutableMapping[
        NonTerminal, Set[GrammarRule]
    ] = filter_grammar(grammar.rules, input)

    earley_table: EarleyChart = EarleyChart(
        semiring,
        EarleyItem(
            SentencePosition(0),
            root_rule,
            DotPosition(1),
            SentencePosition(N - 1),  # N-1 because N-1 is length of input
        ),
    )

    # Initialize
    #
    # ---------------------------------
    # [0, ROOT -> __, 0](R(ROOT -> __))
    root_earley_item: EarleyItem = EarleyItem(
        SentencePosition(0), root_rule, DotPosition(0), SentencePosition(0)
    )
    assert semiring.root_element is not None
    earley_table.add_element(root_earley_item, semiring.root_element)

    for k in range(N):
        j = SentencePosition(k)
        processing_queue: List[EarleyItem] = list(
            earley_table.get_chart_by_end_pos(SentencePosition(k)).keys()
        )
        completion_queue: List[EarleyItem] = []
        ready_for_completes = False

        while len(processing_queue) > 0:
            item = processing_queue.pop(0)
            assert item.end_pos == k
            element = earley_table[item]
            # Complete
            # [j, B -> gamma DOT, k]  [i, A -> alpha DOT B beta, j]
            # -----------------------------------------------------
            #            [i, A -> alpha B DOT beta, k]
            if item.is_done:
                if not ready_for_completes:
                    completion_queue.append(item)
                else:
                    for (prev_item, prev_element) in earley_table.get_chart_by_end_pos(
                        SentencePosition(item.origin_pos)
                    ).items():
                        if (not prev_item.is_done) and prev_item.rule.src[
                            prev_item.dot_idx
                        ] == item.rule.nt:
                            new_completed_item: EarleyItem = EarleyItem(
                                prev_item.origin_pos,
                                prev_item.rule,
                                DotPosition(prev_item.dot_idx + 1),
                                j,
                            )
                            earley_table.add_element(
                                new_completed_item, prev_element.mul(element)
                            )
                            processing_queue.append(new_completed_item)

            # Predict
            #     R(B -> gamma)
            # ---------------------- [i, A -> alpha DOT B beta, k]
            # [k, B -> DOT gamma, k]
            elif item.rule.src[item.dot_idx] in potentially_relevant_rules:
                for new_predicted_item in item.predict(potentially_relevant_rules, j):
                    if new_predicted_item in earley_table.chart:
                        continue
                    earley_table.add_element(
                        new_predicted_item,
                        semiring.initialize_element_from_rule(new_predicted_item.rule),
                    )
                    processing_queue.append(new_predicted_item)

            # Scan
            #  [i, A -> alpha DOT w_k beta, k]
            # ---------------------------------
            # [i, A -> alpha w_k DOT beta, k+1]
            elif item.rule.src[item.dot_idx] == input[k]:
                new_scanned_item = item.scan(j)
                earley_table.add_element(new_scanned_item, element)

            if len(processing_queue) == 0:
                ready_for_completes = True
                processing_queue.extend(completion_queue)
                completion_queue = []

    return earley_table.get_goal_state()


def earley_recognition(
    p_grammar: ProbabilisticGrammar, input: List[Terminal]
) -> Element:
    bool_semiring = BooleanSemiring(p_grammar)
    return earley_semiring(bool_semiring, input)


def earley_viterbi(p_grammar: ProbabilisticGrammar, input: List[Terminal]) -> Element:
    viterbi_semiring = ViterbiSemiring(p_grammar)
    return earley_semiring(viterbi_semiring, input)


def earley_inside(p_grammar: ProbabilisticGrammar, input: List[Terminal]) -> Element:
    inside_semiring = InsideSemiring(p_grammar)
    return earley_semiring(inside_semiring, input)
