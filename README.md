# Pseudo Random Number Generators 

This folder contains the code to create the pseudo random generators, perform statistical tests on them & compare their runtime efficiency. 

PRNG Implemented - 
1. Linear Congruential Generator 
    - Desert Island Generator
    - RANDU Generator
2. Mid-Squared Generator
3.Tausworth Generator
4. Mersenne Twister Generator

Statistical Test Implemented -
1.	Goodness of Fit Tests -  are the PRNs approximately Unif (0,1)?
      - Chi-Square Frequency Test 
      - Kolmogorov-Smirnov Test 
2.	Independence Tests - are the PRNs approximately independent? 
    - Runs Test - Up & Down 
    - Runs Test - Above & Below Mean
    - Autocorrelation Test for independence
  
Here is a brief description of each file:

- *generators.py* - contains the code for implementation of all the generators and the statistical tests mentioned in the report
 
- *test_generators.py* - code that includes the function calls to the generators, statistical tests. This code compares the generators based on their results of tests 
and run time efficiency. Also, includes the code for 3D plots of all the generators

To run the code - 
We need to run the file *test_generators.py*. This will ask for 2 user inputs i.e., 
1. seed 
2. number of iterations (# of random numbers to be generated)

Default confidence level for all the tests is 95%. If we need to change this, we can easily do so in the `main` function of 'test_generators.py'

Once the user inputs are given, the code will show the test results, run time comparison & plots for the generators. 

## Python libraries
We used Python 3.8 and the following libraries for this project:
- numpy
- pandas
- matplotlib
- plotly
- timeit
- random 
- scipy.stats
- itertools
