import itertools
import os
import random
import numpy as np
import torch
import time

from collections.abc import Mapping, Sequence, MutableMapping
from typing import Tuple, Dict, List, Set, Union

from earley import (
    earley_recognition,
    earley_viterbi,
    earley_inside,
)
from semiring.inside import inside_matmul
from semiring.viterbi import viterbi_matmul
from grammar.grammar import (
    create_grammar_rule,
    create_grammar,
    ProbabilisticGrammar,
    GrammarRule,
    Derivation,
    Score,
)
from grammar.tokenizer import Tokenizer
from fast_earley import EarleySupport, earley_semiring_parallelize


def simple_grammar_nonunary() -> Tuple[ProbabilisticGrammar, Tokenizer]:
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../grammar_induction/models/tokenizer")

    grammar: Mapping[
        str,
        List[Dict[str, Union[str, Mapping[int, Tuple[int, int]]]]],
    ] = {
        "p": [{"src": "s", "tgt": "s", "align": {0: (0, 0)}}],
        "s": [
            {
                "src": "s + m",
                "tgt": "s plus m",
                "align": {
                    0: (0, 0),
                    1: (2, 2),
                },
            },
            {"src": "m + 1", "tgt": "m + 1", "align": {0: (0, 0)}},
            {
                "src": "s + b",
                "tgt": "s plus b",
                "align": {
                    0: (0, 0),
                    1: (2, 2),
                },
            },
            {"src": "b + 1", "tgt": "b + 1", "align": {0: (0, 0)}},
        ],
        "m": [
            {
                "src": "m * t",
                "tgt": "product of m and t",
                "align": {0: (0, 2), 1: (2, 4)},
            },
            {
                "src": "t + t",
                "tgt": "sum of t and t",
                "align": {0: (0, 2), 1: (2, 4)},
            },
        ],
        "b": [
            {
                "src": "b * t",
                "tgt": "product of b and t",
                "align": {0: (0, 2), 1: (2, 4)},
            },
            {
                "src": "t + t",
                "tgt": "sum of t and t",
                "align": {0: (0, 2), 1: (2, 4)},
            },
        ],
        "t": [
            {"src": "1", "tgt": "one", "align": {}},
            {"src": "2", "tgt": "two", "align": {}},
            {"src": "3", "tgt": "three", "align": {}},
            {"src": "4", "tgt": "four", "align": {}},
        ],
    }

    tokenizer = Tokenizer.create_tokenizer_from_grammar_dict(grammar)

    nonterminals = {tokenizer.convert_to_tokens([k])[0] for k in grammar.keys()}
    src_terminals = set()
    tgt_terminals = set()
    rules: MutableMapping[int, Set[GrammarRule]] = {}
    num_rules = 0
    root = tokenizer.convert_to_tokens(["p"])[0]
    for nt, rhs_options in grammar.items():
        nt_tok = tokenizer.convert_to_tokens([nt])[0]
        for rhs in rhs_options:
            src_string = rhs["src"]
            tgt_string = rhs["tgt"]
            assert type(src_string) == str
            assert type(tgt_string) == str
            src_tok = tokenizer.convert_to_tokens(src_string.split())
            tgt_tok = tokenizer.convert_to_tokens(tgt_string.split())
            src_ts = list(src_tok)
            tgt_ts = list(tgt_tok)
            assert isinstance(rhs["align"], Mapping)
            for (src_nt_idx, tgt_nt_idx) in rhs["align"].values():
                src_ts[src_nt_idx] = None
                tgt_ts[tgt_nt_idx] = None
            src_terminals.update(set(src_ts))
            tgt_terminals.update(set(tgt_ts))
            if nt_tok not in rules:
                rules[nt_tok] = {
                    create_grammar_rule(
                        nt_tok, src_tok, tgt_tok, rhs["align"], num_rules
                    )
                }

            else:
                rules[nt_tok].add(
                    create_grammar_rule(
                        nt_tok, src_tok, tgt_tok, rhs["align"], num_rules
                    )
                )
            num_rules += 1
    if None in src_terminals:
        src_terminals.remove(None)
    if None in tgt_terminals:
        tgt_terminals.remove(None)
    grammar_object = create_grammar(
        nonterminals, src_terminals, tgt_terminals, rules, root
    )

    return (
        ProbabilisticGrammar.create_random_parameter(grammar_object, in_logspace=False),
        tokenizer,
    )


def simple_grammar_nonunary_logprob() -> Tuple[ProbabilisticGrammar, Tokenizer]:
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "../grammar_induction/models/tokenizer")

    grammar: Mapping[
        str,
        List[Dict[str, Union[str, Mapping[int, Tuple[int, int]]]]],
    ] = {
        "p": [{"src": "s", "tgt": "s", "align": {0: (0, 0)}}],
        "s": [
            {
                "src": "s + m",
                "tgt": "s plus m",
                "align": {
                    0: (0, 0),
                    1: (2, 2),
                },
            },
            {"src": "m + 1", "tgt": "m + 1", "align": {0: (0, 0)}},
            {
                "src": "s + b",
                "tgt": "s plus b",
                "align": {
                    0: (0, 0),
                    1: (2, 2),
                },
            },
            {"src": "b + 1", "tgt": "b + 1", "align": {0: (0, 0)}},
        ],
        "m": [
            {
                "src": "m * t",
                "tgt": "product of m and t",
                "align": {0: (0, 2), 1: (2, 4)},
            },
            {
                "src": "t + t",
                "tgt": "sum of t and t",
                "align": {0: (0, 2), 1: (2, 4)},
            },
        ],
        "b": [
            {
                "src": "b * t",
                "tgt": "product of b and t",
                "align": {0: (0, 2), 1: (2, 4)},
            },
            {
                "src": "t + t",
                "tgt": "sum of t and t",
                "align": {0: (0, 2), 1: (2, 4)},
            },
        ],
        "t": [
            {"src": "1", "tgt": "one", "align": {}},
            {"src": "2", "tgt": "two", "align": {}},
            {"src": "3", "tgt": "three", "align": {}},
            {"src": "4", "tgt": "four", "align": {}},
        ],
    }

    tokenizer = Tokenizer.create_tokenizer_from_grammar_dict(grammar)

    nonterminals = {tokenizer.convert_to_tokens([k])[0] for k in grammar.keys()}
    src_terminals = set()
    tgt_terminals = set()
    rules: MutableMapping[int, Set[GrammarRule]] = {}
    num_rules = 0
    root = tokenizer.convert_to_tokens(["p"])[0]
    for nt, rhs_options in grammar.items():
        nt_tok = tokenizer.convert_to_tokens([nt])[0]
        for rhs in rhs_options:
            src_string = rhs["src"]
            tgt_string = rhs["tgt"]
            assert type(src_string) == str
            assert type(tgt_string) == str
            src_tok = tokenizer.convert_to_tokens(src_string.split())
            tgt_tok = tokenizer.convert_to_tokens(tgt_string.split())
            src_ts = list(src_tok)
            tgt_ts = list(tgt_tok)
            assert isinstance(rhs["align"], Mapping)
            for (src_nt_idx, tgt_nt_idx) in rhs["align"].values():
                src_ts[src_nt_idx] = None
                tgt_ts[tgt_nt_idx] = None
            src_terminals.update(set(src_ts))
            tgt_terminals.update(set(tgt_ts))
            if nt_tok not in rules:
                rules[nt_tok] = {
                    create_grammar_rule(
                        nt_tok, src_tok, tgt_tok, rhs["align"], num_rules
                    )
                }

            else:
                rules[nt_tok].add(
                    create_grammar_rule(
                        nt_tok, src_tok, tgt_tok, rhs["align"], num_rules
                    )
                )
            num_rules += 1
    if None in src_terminals:
        src_terminals.remove(None)
    if None in tgt_terminals:
        tgt_terminals.remove(None)
    grammar_object = create_grammar(
        nonterminals, src_terminals, tgt_terminals, rules, root
    )

    return ProbabilisticGrammar.create_random_parameter(grammar_object), tokenizer


########################## SERIAL PROCESSING ##########################


def test_earley_bool_semiring(length=4):
    # p_grammar, tokenizer = simple_grammar_nonunary_logprob()
    p_grammar, tokenizer = simple_grammar_nonunary()
    grammar = p_grammar.grammar

    random_strings = set()

    for l in range(1, length + 1):
        for t in itertools.product(grammar.src_terminals, repeat=l):
            random_strings.add(tokenizer.decode(list(t)))

    parseable = set()
    for d in grammar.derivation_generator(grammar.root_nt, length):
        parseable.add(d.get_produced_src_string(tokenizer))
    should_fail = random_strings - parseable
    
    times = []
    for src in parseable:
        src_tok = tokenizer.convert_to_tokens(src.split())
        start = time.time()
        was_parsed = earley_recognition(p_grammar, src_tok)
        times.append(time.time() - start)
        assert (
            was_parsed.get_score()
        ), f"{src} should be parseable by the grammar but it is not."
    print("====== passed parseable =====")

    for src in should_fail:
        src_tok = tokenizer.convert_to_tokens(src.split())
        start = time.time()
        was_parsed = earley_recognition(p_grammar, src_tok)
        times.append(time.time() - start)
        assert (
            not was_parsed.get_score()
        ), f"{src} should not be parseable by the grammar but it is."
    print("====== passed should-not-be-parseable =====")
    return sum(times) / len(times)

def test_earley_viterbi_semiring(run_times=25):
    p_grammar, tokenizer = simple_grammar_nonunary_logprob()
    # p_grammar, tokenizer = simple_grammar_nonunary()

    times = []
    already_parsed = set()
    for (d, score) in p_grammar.top_down_generator(in_logspace=True):
        src = d.get_produced_src_string(tokenizer)
        if src in already_parsed:
            continue
        src_tok = tokenizer.convert_to_tokens(src.split())
        start = time.time()
        earley_result = earley_viterbi(p_grammar, src_tok)
        times.append(time.time() - start)
        assert np.isclose(
            score, earley_result.get_score()[0]
        ), f"Expected a score of {score} but got {earley_result.get_score()[0]}"
        derivation_match = False
        for cand_d in earley_result.get_score()[1]:
            if cand_d.equals(d):
                derivation_match = True
                break
        assert (
            derivation_match
        ), f"`{src}` did not yield the correct viterbi derivation."
        already_parsed.add(src)
        run_times -= 1
        if run_times < 0:
            break
    print("====== passed viterbi =====")
    return sum(times) / len(times)


def test_earley_inside_semiring(num_src_to_check=50):
    p_grammar, tokenizer = simple_grammar_nonunary_logprob()
    # p_grammar, tokenizer = simple_grammar_nonunary()
    semiring_zero = np.log(1e-10)
    semiring_add = np.logaddexp

    strings_to_check: Dict[str, Set[Tuple[Derivation, Score]]] = {}

    for (d, score) in p_grammar.top_down_generator(num_src_to_check*4, in_logspace=True):
        src = d.get_produced_src_string(tokenizer)
        if len(strings_to_check) > num_src_to_check and src not in strings_to_check:
            continue
        if src in strings_to_check:
            strings_to_check[src].add((d, score))
        else:
            strings_to_check[src] = {(d, score)}
    
    times = []
    for src, derivs in strings_to_check.items():
        src_tok = tokenizer.convert_to_tokens(src.split())
        total_deriv_score = semiring_zero
        for d in derivs:
            total_deriv_score = semiring_add(total_deriv_score, d[1])
        start = time.time()
        earley_result = earley_inside(p_grammar, src_tok)
        times.append(time.time() - start)

        assert np.isclose(
            total_deriv_score, earley_result.get_score()
        ), f"Expected a score of {total_deriv_score} but got {earley_result.get_score()}"

    print("====== passed inside =====")
    return sum(times) / len(times)


########################## PARALLEL PROCESSING ##########################


def test_earley_parallelized_inside(num_src_to_check=50):
    p_grammar, tokenizer = simple_grammar_nonunary()
    # p_grammar, tokenizer = simple_grammar_nonunary_logprob()

    inside_add = torch.add
    inside_mul = torch.mul
    inside_zero = 0
    inside_one = 1
    # inside_add = torch.logaddexp
    # inside_mul = torch.add
    # inside_zero = torch.tensor(-1e10) # float('-inf')
    # inside_one = 0
    support = EarleySupport.initialize_parallelization_matrices(
        p_grammar, tokenizer, inside_zero
    )

    strings_to_check: Dict[str, Set[Tuple[Derivation, Score]]] = {}

    for (d, score) in p_grammar.top_down_generator(
        max_num_generated=num_src_to_check*4, in_logspace=False
    ):
        src = d.get_produced_src_string(tokenizer)
        if len(strings_to_check) > num_src_to_check and src not in strings_to_check:
            continue
        if src in strings_to_check:
            strings_to_check[src].add((d, score))
        else:
            strings_to_check[src] = {(d, score)}
    
    times = []
    for src, derivs in strings_to_check.items():
        src_tok = tokenizer.convert_to_tokens(src.split())
        total_deriv_score = inside_zero
        for d, score in derivs:
            total_deriv_score = inside_add(total_deriv_score, torch.tensor(score))
        start = time.time()
        inside_value = earley_semiring_parallelize(
            src_tok,
            support,
            inside_add,
            inside_matmul,
            inside_mul,
            inside_zero,
        )
        times.append(time.time() - start)

        assert np.isclose(
            total_deriv_score, inside_value
        ), f"Expected a score of {total_deriv_score} but got {inside_value}"

    print("====== passed inside parallelize =====")
    return sum(times) / len(times)

def test_earley_parallelized_viterbi(run_times=25):
    p_grammar, tokenizer = simple_grammar_nonunary()
    # p_grammar, tokenizer = simple_grammar_nonunary_logprob()

    viterbi_add = torch.maximum
    viterbi_mul = torch.mul
    viterbi_zero = 0
    viterbi_one = 1
    # viterbi_add = torch.maximum
    # viterbi_mul = torch.add
    # viterbi_zero = -1e10
    # viterbi_one = 0
    support = EarleySupport.initialize_parallelization_matrices(
        p_grammar, tokenizer, viterbi_zero
    )

    already_parsed = set()
    times = []
    for (d, score) in p_grammar.top_down_generator(in_logspace=False):
        src = d.get_produced_src_string(tokenizer)
        if src in already_parsed:
            continue
        src_tok = tokenizer.convert_to_tokens(src.split())
        start = time.time()
        viterbi_score = earley_semiring_parallelize(
            src_tok,
            support,
            viterbi_add,
            viterbi_matmul,
            viterbi_mul,
            viterbi_zero,
        )
        times.append(time.time() - start)
        assert np.isclose(
            score, viterbi_score
        ), f"Expected a score of {score} but got {viterbi_score}"
        already_parsed.add(src)
        run_times -= 1
        if run_times < 0:
            break

    print("====== passed viterbi parallelize =====")
    return sum(times) / len(times)
