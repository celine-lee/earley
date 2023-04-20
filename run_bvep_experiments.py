from tests.test_earley import *
import time

# Grammar options: simple_grammar_nonunary, simple_grammar_nonunary_logprob

# Semirings: bool, inside, viterbi

# Algorithms: serial, bv

# randomly generates up-to-length strings from simple_grammar_nonunary_logprob
# then parses them, ensuring that passables ones pass and unpassable ones don't. 
# returns average time of parsing them
serial_bool_simple = test_earley_bool_semiring(length=4)

# randomly generates run_times strings from simple_grammar_nonunary_logprob
# then parses them, ensuring that viterbi scores parsed are the scores expected
# returns average time of parsing them
serial_viterbi_simple = test_earley_viterbi_semiring(run_times=25)
# simple_grammar_nonunary() oh we should make sure this is logprob TODO
bv_viterbi_simple =  test_earley_parallelized_viterbi(run_times=25)

# randomly generates num_src_to_check strings from simple_grammar_nonunary_logprob
# then parses them, ensuring that inside scores parsed are the scores expected
# returns average time of parsing them
serial_inside_simple = test_earley_inside_semiring(num_src_to_check=50)
# simple_grammar_nonunary() oh we should make sure this is logprob TODO
bv_inside_simple = test_earley_parallelized_inside(num_src_to_check=50)


print('*** Runtimes ***')
print('Semiring\tGrammar\tSerial\tBV')
print(f'Bool\t\tsimple_logprob\t{serial_bool_simple:.3f}\tNA')
print(f'Viterbi\t\tsimple_logprob\t{serial_viterbi_simple:.3f}\t{bv_viterbi_simple:.3f}')
print(f'Inside\t\tsimple_logprob\t{serial_inside_simple:.3f}\t{bv_inside_simple:.3f}')
