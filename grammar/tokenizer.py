from __future__ import annotations

from typing import Dict, List, Union, Tuple


class Tokenizer:
    def __init__(self, vocab_mapping: Dict[str, int], unk_tok: int):
        self.str_to_token = vocab_mapping
        self.token_to_str = {v: k for k, v in vocab_mapping.items()}
        assert unk_tok in self.token_to_str
        self.UNK_TOK = unk_tok
        self.UNK = self.token_to_str[unk_tok]

    @staticmethod
    def create_tokenizer_from_grammar_dict(
        grammar: Dict[
            str,
            List[Dict[str, Union[str, Dict[int, Tuple[List[int], List[int]]]]]],
        ]
    ) -> Tokenizer:
        tokenizer: Dict[str, int] = {"<unk>": 0}

        for nt, rules in grammar.items():
            if nt not in tokenizer:
                tokenizer[nt] = len(tokenizer)
            for rule in rules:
                src_string = rule["src"]
                assert type(src_string) == str
                for s in src_string.split():
                    if s not in tokenizer:
                        tokenizer[s] = len(tokenizer)
                tgt_string = rule["tgt"]
                assert type(tgt_string) == str
                for t in tgt_string.split():
                    if t not in tokenizer:
                        tokenizer[t] = len(tokenizer)

        return Tokenizer(tokenizer, 0)

    def convert_to_tokens(self, input: List[str]) -> List[int]:
        token_ids = []
        for tok in input:
            if tok in self.str_to_token:
                token_ids.append(self.str_to_token[tok])
            else:
                token_ids.append(self.UNK_TOK)
        return token_ids

    def convert_ids_to_tokens(self, input: List[int]) -> List[str]:
        return [self.token_to_str[tok] for tok in input]

    def decode(self, tokenized_input: List[int] | int) -> str:
        if type(tokenized_input) == int:
            return self.token_to_str[tokenized_input]
        assert (
            type(tokenized_input) == list
        ), f"input should be type list, but is {type(tokenized_input)}"
        return " ".join([self.token_to_str[tok] for tok in tokenized_input])

    def __len__(self):
        return len(self.str_to_token)
