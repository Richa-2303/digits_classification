import itertools
def test_for_hyper_param_combinations_count():
    param_type=['gamma','C']
    gamma_ranges=[0.001,0.01,0.1,1,10,100]
    C_ranges=[0.1,1,2,5,10]
    list_of_all_param_combination = [dict(zip(param_type, values)) for values in itertools.product(gamma_ranges, C_ranges)]
    assert len(list_of_all_param_combination) == len(gamma_ranges)*len(C_ranges)

def test_for_hyper_param_combinations_values():
    param_type=['gamma','C']
    gamma_ranges=[0.001,0.01,0.1,1,10,100]
    C_ranges=[0.1,1,2,5,10]
    list_of_all_param_combination = [dict(zip(param_type, values)) for values in itertools.product(gamma_ranges, C_ranges)]
    expected_param_combo1={'gamma':0.001,'C':1}
    expected_param_combo2={'gamma':0.01,'C':1}
    assert (expected_param_combo1 in list_of_all_param_combination) and (expected_param_combo2 in list_of_all_param_combination)
