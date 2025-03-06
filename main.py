from TSPD.LNS_final_Greedy import main
import matplotlib.pyplot as plt
import pandas as pd

# main()
choose = int(input('''
                1. Draw 50 cities
                2. Draw 100 cities
                3. Draw 200 cities
                   '''))
if choose == 1:
    data = pd.read_csv('Result/output_50.csv')
    data_greedy = pd.read_csv('Result/output_greedy_50.csv')
    data_alns = pd.read_csv('Result/alns_50.csv')
    save_path = 'Result/Images/compare_50.png'
    title = 'Fitness 50 cities'
elif choose == 2:
    data = pd.read_csv('Result/output_100.csv')
    data_greedy = pd.read_csv('Result/output_greedy_100.csv')
    data_alns = pd.read_csv('Result/alns_100.csv')
    save_path = 'Result/Images/compare_100.png'
    title = 'Fitness 100 cities'
elif choose == 3:
    data = pd.read_csv('Result/output_200.csv')
    data_greedy = pd.read_csv('Result/output_greedy_200.csv')
    data_alns = pd.read_csv('Result/alns_200.csv')
    save_path = 'Result/Images/compare_200.png'
    title = 'Fitness 200 cities'
else: 
    print('Please choose again')
alns = data_alns.iloc[:,13]
fitness = data.iloc[:,21]
greedy = data_greedy.iloc[:,21]
plt.plot(fitness,color='blue',label='LNS')
plt.plot(greedy,color='red',label='LNS_greedy')
plt.plot(alns,color='green',label='ALNS')

plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title(f'{title}')
plt.legend()
plt.savefig(f'{save_path}')
plt.show()

print('done ')