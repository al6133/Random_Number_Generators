import pandas as pd
import numpy as np
import random
import scipy.stats as st
from scipy.stats import chi2
from itertools import groupby

""""""
""" 
I will be focussing on below 5 generators -
 1. Linear Congruential Genrators are of the form -
   X[i] = (aX[i-1] + c)mod m here X[0] is the seed
   R[i] = X[i]/m
     1.i.  Desert Island Generator
     1.ii. RANDU
 2. Merrene Twister - This is the one that's used in python random library
 3. Mid Squared Generator
 4. Tausworth Generator
"""

#Linear Congruential Generator
def lcg(a, c, m, num_iter, seed):
    # Initialize variables
    value_x = seed
    random_list = []
    for i in range(num_iter):
        # Calculate X[i] value
        value_x = (a * value_x + c) % m

        # Calculate Random number R[i] =X[i]/ m
        value_r = (value_x / m)

        random_list.append(value_r)
    return random_list

#Mersenne Twister Generator - I have used the inbuild random function in python for this.
def mersenne_twister(num_iter, seed):
    random_list = []
    random.seed(seed)
    for i in range(num_iter):
        random_list.append(random.random())
    return random_list

#Mid Squared Generator
def mid_squared_generator(seed, num_iter):

    value_x = int(seed)
    random_list = []
    len_x = len(str(value_x))

    for i in range(num_iter):
        value_squared = str(value_x ** 2)

        if len(value_squared)< 2*len_x:
            value_squared = value_squared.zfill(2*len_x)
        start = int(len_x/2)
        end = int(start + len_x)

        value_x = int(value_squared[start:end])
        temp_m = ("1").zfill(len_x+1)
        m = int("".join(reversed(temp_m)))

        value_r = value_x/m

        random_list.append(value_r)
    return random_list

#Tausworth Generator
def tausworth(seed, num_iter, r, q, l):
    random.seed(seed)

    bits = []
    for i in range(q):
        bits.append(random.randint(0, 1))
    bits = np.array(bits)
    l = l  # Number of digits to consider
    q = q
    r = r
    n = num_iter
    list_i = list(range(q, n))

    for i in list_i:
        b_lag_r = bits[i - r]
        b_lag_q = bits[i - q]
        b = b_lag_r ^ b_lag_q
        bits = np.append(bits, b)
    bits = list(map(str, bits))
    urn_base2 = [bits[x:x + l] for x in range(0, len(bits), l)]
    urn_base10 = []
    for i in list(urn_base2):
        urn_base10.append((int("".join(i), 2) / 2 ** l))
    return urn_base10
""""""
""" -----------------------------------Statistical Tests ----------------------------------------------------
Two Classes of Tests-  
1. Goodness of Fit tests -  are the PRNs approximately Unif [0,1)?
    * Chi-Square Frequency Test
    * Kolmogorov-Smirnov Test 
2. Independence Tests - are the PRNs approximately independent?  
    * Runs Test - Up & Down
    * Runs Test - Above & Below Mean
    * Autocorrelation Test for independence
"""
### Goodness of Fit Tests ###

# * H0 - Null Hypothesis - Observations are approx. Uniform U[0,1)
# * H1 - Alternate Hypothesis - Observations are not Uniform

def get_rand_freq(rand_list):
    Keys = list(range(1, 11))
    Values = list(np.zeros(10))
    _rand_freq = dict(zip(Keys, Values))

    for rand in rand_list:
        rand = float(rand)

        if rand < 0.1:
            _rand_freq[1] += 1
        elif rand < 0.2:
            _rand_freq[2] += 1
        elif rand < 0.3:
            _rand_freq[3] += 1
        elif rand < 0.4:
            _rand_freq[4] += 1
        elif rand < 0.5:
            _rand_freq[5] += 1
        elif rand < 0.6:
            _rand_freq[6] += 1
        elif rand < 0.7:
            _rand_freq[7] += 1
        elif rand < 0.8:
            _rand_freq[8] += 1
        elif rand < 0.9:
            _rand_freq[9] += 1
        elif rand < 1.0:
            _rand_freq[10] += 1
    return _rand_freq


def chi_squared_test(random_list, confidence_level):
    observed = get_rand_freq(random_list)

    # Number of Intervals
    k = 10.0

    # Degree of Freedom
    degree_of_freedom = k - 1

    # Chi square critical value for given confidence level
    _critical_value = chi2.ppf(confidence_level, degree_of_freedom)

    expected = len(random_list) / k

    _chi_square = 0.0
    for key in observed:
        _chi_square += ((expected - observed[key]) ** 2) / expected

    result = "Accept"

    if _chi_square > _critical_value:
        result = "Reject"
    print("Chi Square: ", _chi_square)
    print("Crit Value: ", _critical_value)
    print("Result is: ", result)

    return result


def kolmogorov_smirnov_test(random_list, confidence_level):
    d_plus = []
    d_minus = []
    num = len(random_list)

    # Rank the random numbers
    random_list_sorted = sorted(random_list)

    # Calculate D_plus using the formula D+ = max(i/N - R(i))
    for i in range(1, num + 1):
        val_plus = i / num - random_list_sorted[i - 1]
        d_plus.append(val_plus)
    d_plus_max = max(d_plus)

    # Calculate D_minus using the formula D- = max(R(i) - (i -1)/n)
    for i in range(1, num + 1):
        val_minus = random_list_sorted[i - 1] - ((i - 1) / num)
        d_minus.append(val_minus)
    d_minus_max = max(d_minus)

    d_stat = max(d_plus_max, d_minus_max)

    critical_value = 0.0
    alpha = round(1 - confidence_level, 2)
    if alpha == 0.1:
        critical_value = 1.22 / np.sqrt(num)
    elif alpha == 0.05:
        critical_value = 1.36 / np.sqrt(num)
    elif alpha == 0.01:
        critical_value = 1.63 / np.sqrt(num)

    result = "Accept"

    if d_stat > critical_value:
        result = "Reject"
    print("d_stat: ", d_stat)
    print("Critical Value: ", critical_value)
    print("Result is: ", result)

    return result

### Independence Tests ###

# * H0 - Null Hypothesis - Observations are approx. independent U[0,1)
# * H1 - Alternate Hypothesis - Observations are not independent

def runs_up_down_test(random_list, confidence_level):
    runs = []
    runs_length = []
    run_dir = 'none'
    num = len(random_list)

    for i in range(num - 1):
        cur_val = random_list[i]
        next_val = random_list[i + 1]

        if cur_val == next_val:
            run_dir = run_dir

        elif cur_val < next_val:
            run_dir = '+'
        else:
            run_dir = '-'
        runs.append(run_dir)

    A = len([sum(1 for _ in r) for _, r in groupby(runs)])

    mean = ((2 * num - 1) / 3)
    variance = ((16 * num - 29) / 90)
    z_stat = (A - mean) / np.sqrt(variance)

    result = "Accept"
    alpha = 1 - confidence_level

    critical_value = st.norm.ppf(1 - alpha / 2)  # 1-alpha  to get the positive alpha value

    if abs(z_stat) > critical_value:
        result = "Reject"
    print("Z_stat: ", abs(z_stat))
    print("Critical Value: ", critical_value)
    print("Result is: ", result)

    return result

def runs_above_below_mean(random_list, confidence_level):

    runs = []
    num = len(random_list)

    for i in range(num):
        cur_val = random_list[i]

        if cur_val >= 0.5:
            run_dir = '+'
        else :
            run_dir ='-'
        runs.append(run_dir)

    B = len([sum(1 for _ in r) for _, r in groupby(runs)])

    n1 = runs.count('+')
    n2 = runs.count('-')

    mean = ((2*n1*n2)/num + 1/2)
    variance = (2*n1*n2*(2*n1*n2 - num))/((num**2)*(num-1))
    z_stat = (B - mean)/np.sqrt(variance)

    result = "Accept"
    alpha = 1 - confidence_level

    critical_value = st.norm.ppf(1-alpha/2)  #1-alpha  to get the positive alpha value

    if abs(z_stat) > critical_value:
        result = "Reject"
    print ("Z_stat: ", abs(z_stat))
    print ("Critical Value: ", critical_value)
    print ("Result is: ", result)

    return result

def autocorrelation_test(random_list, confidence_level):
    n = len(random_list)
    rho_hat = 0.0
    for i in range(len(random_list)-1):

        cur_val = random_list[i]
        next_val = random_list[i+1]
        rho_hat = rho_hat + (cur_val*next_val)
    rho_hat = ((12/(n-1))*rho_hat)-3
    variance = (13*n - 19)/(n-1)**2
    z_stat =  rho_hat/np.sqrt(variance)
    result = "Accept"
    alpha = 1 - confidence_level
    critical_value = st.norm.ppf(1-alpha/2)  #1-alpha  to get the positive alpha value

    if abs(z_stat) > critical_value:
        result = "Reject"
    print ("Z_stat: ", abs(z_stat))
    print ("Critical Value: ", critical_value)
    print ("Result is: ", result)
    return result