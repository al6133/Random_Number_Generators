import pandas as pd
import numpy as np
import random
import timeit
import plotly.express as px
from generators import *
pd.options.display.float_format = '{:.8f}'.format


def test_gen(num_iter=100, seed=123.0, confidence_level=0.95):
    print(seed)
    print(num_iter)
    # 1. Desert Island Generator
    desert_island_lcg_list = lcg(a=16807, c=0, m=(2 ** 31 - 1), num_iter=num_iter, seed=seed)

    desert_island_result = ['Desert Island LCG']
    desert_island_result.append(chi_squared_test(desert_island_lcg_list, confidence_level))
    desert_island_result.append(kolmogorov_smirnov_test(desert_island_lcg_list, confidence_level))
    desert_island_result.append(runs_up_down_test(desert_island_lcg_list, confidence_level))
    desert_island_result.append(runs_above_below_mean(desert_island_lcg_list, confidence_level))
    desert_island_result.append(autocorrelation_test(desert_island_lcg_list, confidence_level))

    # 1. RANDU Generator
    RANDU_lcg_list = lcg(a=65539, c=0, m=(2 ** 31), num_iter=num_iter, seed=seed)
    RANDU_result = ['RANDU LCG']
    RANDU_result.append(chi_squared_test(RANDU_lcg_list, confidence_level))
    RANDU_result.append(kolmogorov_smirnov_test(RANDU_lcg_list, confidence_level))
    RANDU_result.append(runs_up_down_test(RANDU_lcg_list, confidence_level))
    RANDU_result.append(runs_above_below_mean(RANDU_lcg_list, confidence_level))
    RANDU_result.append(autocorrelation_test(RANDU_lcg_list, confidence_level))

    # Mersenne Twister
    mersenne_twister_list = mersenne_twister(seed=seed, num_iter=num_iter)
    mersenne_result = ['Mersenne Twister']
    mersenne_result.append(chi_squared_test(mersenne_twister_list, confidence_level))
    mersenne_result.append(kolmogorov_smirnov_test(mersenne_twister_list, confidence_level))
    mersenne_result.append(runs_up_down_test(mersenne_twister_list, confidence_level))
    mersenne_result.append(runs_above_below_mean(mersenne_twister_list, confidence_level))
    mersenne_result.append(autocorrelation_test(mersenne_twister_list, confidence_level))

    print('Mid Squared')
    # Mid Squared Generator
    mid_squared_list = mid_squared_generator(seed=seed, num_iter=num_iter)
    mid_squared_result = ['Mid Squared']
    mid_squared_result.append(chi_squared_test(mid_squared_list, confidence_level))
    mid_squared_result.append(kolmogorov_smirnov_test(mid_squared_list, confidence_level))
    mid_squared_result.append(runs_up_down_test(mid_squared_list, confidence_level))
    mid_squared_result.append(runs_above_below_mean(mid_squared_list, confidence_level))
    mid_squared_result.append(autocorrelation_test(mid_squared_list, confidence_level))

    # Tausworth Generator
    tausworth_list = tausworth(seed=seed, num_iter=num_iter, r=30, q=35, l=4)
    tausworth_result = ['Tausworth']
    tausworth_result.append(chi_squared_test(tausworth_list, confidence_level))
    tausworth_result.append(kolmogorov_smirnov_test(tausworth_list, confidence_level))
    tausworth_result.append(runs_up_down_test(tausworth_list, confidence_level))
    tausworth_result.append(runs_above_below_mean(tausworth_list, confidence_level))
    tausworth_result.append(autocorrelation_test(tausworth_list, confidence_level))

    colnames = ['PRN', 'Chi-Squared', 'KS', 'Runs_Up_Down', 'Runs_Above_Below_Mean', 'Autocorrelation']
    result_list = [desert_island_result, RANDU_result, mersenne_result, mid_squared_result, tausworth_result]
    results = pd.DataFrame(result_list, columns=colnames)

    return results, desert_island_lcg_list, RANDU_lcg_list, mersenne_twister_list, mid_squared_list, tausworth_list

#Compare the run time efficiency of Pseudo Random Generators
def test_runtime(sample_size):
    iterations = sample_size
    di_time = ['Desert Island LCG']
    randu_time = ['RANDU LCG']
    mt_time = ['Mersenne Twister']
    ms_time = ['Mid Squared']
    taus_time = ['Tausworth']

    for num in iterations:
        beg = timeit.default_timer()
        lcg(a=16807, c=0, m=(2 ** 31 - 1), num_iter=num, seed=seed)
        end = timeit.default_timer()
        di_time.append(round(end - beg, 4))

        TotTime = []
        beg = timeit.default_timer()
        lcg(a=65539, c=0, m=(2 ** 31), num_iter=num, seed=seed)
        end = timeit.default_timer()
        randu_time.append(round(end - beg, 4))

        beg = timeit.default_timer()
        mersenne_twister(seed=seed, num_iter=num)
        end = timeit.default_timer()
        mt_time.append(round(end - beg, 4))

        beg = timeit.default_timer()
        mid_squared_generator(seed=seed, num_iter=num)
        end = timeit.default_timer()
        ms_time.append(round(end - beg, 4))

        beg = timeit.default_timer()
        tausworth(seed=seed, num_iter=num, r=30, q=35, l=4)
        end = timeit.default_timer()
        taus_time.append(round(end - beg, 4))

    iterations.insert(0, 'PRN')
    time_taken = [di_time, randu_time, mt_time, ms_time, taus_time]
    results_time = pd.DataFrame(time_taken, columns=iterations)
    return results_time


if __name__ == "__main__":

    seed = float(input("Enter valid seed value: "))
    num_iter = int(input("Enter Number of Random Numbers to be generated "))

    confidence_level = 0.95

    results, desert_island_lcg_list, RANDU_lcg_list, mersenne_twister_list, mid_squared_list, tausworth_list = test_gen(
        num_iter=num_iter, seed=seed, confidence_level=confidence_level)
    results_time = test_runtime(sample_size=[100, 500, 1000, 10000, 100000])
    print(results)
    print(results_time)
    print()

    print('Desert Island Generator with seed {} and iterations {}'.format(seed, num_iter))

    x = pd.Series(desert_island_lcg_list[0: len(desert_island_lcg_list) - 2])
    y = pd.Series(desert_island_lcg_list[1: len(desert_island_lcg_list) - 1])
    z = pd.Series(desert_island_lcg_list[2: len(desert_island_lcg_list)])
    data = pd.concat([x, y, z], axis=1).reset_index(drop=True)
    data.columns = ['x', 'y', 'z']
    fig = px.scatter_3d(data, x='x', y='y', z='z', width=1000, height=500)
    fig.show()

    print('RANDU Generator with seed {} and iterations {}'.format(seed, num_iter))

    x = pd.Series(RANDU_lcg_list[0: len(RANDU_lcg_list) - 2])
    y = pd.Series(RANDU_lcg_list[1: len(RANDU_lcg_list) - 1])
    z = pd.Series(RANDU_lcg_list[2: len(RANDU_lcg_list)])
    data = pd.concat([x, y, z], axis=1).reset_index(drop=True)
    data.columns = ['x', 'y', 'z']
    fig = px.scatter_3d(data, x='x', y='y', z='z', width=1000, height=500)
    fig.show()

    print('Mersenne Twister Generator with seed {} and iterations {}'.format(seed, num_iter))

    x = pd.Series(mersenne_twister_list[0: len(mersenne_twister_list) - 2])
    y = pd.Series(mersenne_twister_list[1: len(mersenne_twister_list) - 1])
    z = pd.Series(mersenne_twister_list[2: len(mersenne_twister_list)])
    data = pd.concat([x, y, z], axis=1).reset_index(drop=True)
    data.columns = ['x', 'y', 'z']
    fig = px.scatter_3d(data, x='x', y='y', z='z', width=1000, height=450)
    fig.show()

    print('Mid Squared Generator with seed {} and iterations {}'.format(seed, num_iter))
    x = pd.Series(mid_squared_list[0: len(mid_squared_list) - 2])
    y = pd.Series(mid_squared_list[1: len(mid_squared_list) - 1])
    z = pd.Series(mid_squared_list[2: len(mid_squared_list)])
    data = pd.concat([x, y, z], axis=1).reset_index(drop=True)
    data.columns = ['x', 'y', 'z']
    fig = px.scatter_3d(data, x='x', y='y', z='z', width=1000, height=450)
    fig.show()

    print('Mid Squared Generator with seed {} and iterations {}'.format(seed, num_iter))
    x = pd.Series(tausworth_list[0: len(tausworth_list) - 2])
    y = pd.Series(tausworth_list[1: len(tausworth_list) - 1])
    z = pd.Series(tausworth_list[2: len(tausworth_list)])
    data = pd.concat([x, y, z], axis=1).reset_index(drop=True)
    data.columns = ['x', 'y', 'z']
    fig = px.scatter_3d(data, x='x', y='y', z='z', width=1000, height=450)
    fig.show()



