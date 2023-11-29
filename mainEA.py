import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
import EA
import DA
import time
import datetime
import pickle
seed = 29
np.random.seed(seed)

save_states = True
use_states = True
start = 70
stop = 100

def getState():  
    return np.random.get_state()

def setState(state_np):
    np.random.set_state(state_np)

tic = time.perf_counter()



'''ES possible inputs'''
## TABLE C
input_functions = [
(0.5,"Add((-round(100*p1),round(100*p2)))"), 
(0.5,"AddElementwise((-round(100*p1),round(100*p2)))"), 
(0.5,"AdditiveGaussianNoise(scale=(0,round(p1*255)),per_channel=p2)"),
(0.5,"AdditiveLaplaceNoise(scale=(0,round(p1*255)),per_channel=p2)"),
(0.5,"AdditivePoissonNoise((0,p1*10),per_channel=p2)"), 
(0.5,"Multiply((2*min([p1,p2]),2*max([p1,p2])))"),
(0.5,"MultiplyElementwise((2*min([p1,p2]),2*max([p1,p2])))"),
(0.5,"Cutout(nb_iterations=round(10*p1),size=p2,fill_mode='gaussian' if p3<0.5 else 'constant')"),
(0.5,"Dropout(p=(0,0.5*p1),per_channel=p2)"),
(0.5,"CoarseDropout(p1,size_percent=p2)"),
(0.5,"ReplaceElementwise(0.5*p1,[0,255],per_channel=p2)"), 
(0.5,"SaltAndPepper(0.5*p1,per_channel=p2)"), 
(0.5,"CoarseSaltAndPepper(0.5*p1,size_percent=p2,per_channel=p3)"),  
(0.5,"Salt(0.5*p1,per_channel=p2)"),
(0.5,"CoarseSalt(0.5*p1,size_percent=p2,per_channel=p3)"),
(0.5,"Pepper(0.5*p1,per_channel=p2)"),
(0.5,"CoarsePepper(0.5*p1,size_percent=p2)"),
(1,"Fliplr(p1)"),
(1,"Flipud(p1)"),
(0.5,"ScaleX((2*min([p1,p2]),2*max([p1,p2])),mode=mode_op[int(5*p3)])"),
(0.5,"ScaleY((2*min([p1,p2]),2*max([p1,p2])),mode=mode_op[int(5*p3)])"),
(0.5,"TranslateX(percent=(-p1,p2),mode=mode_op[int(5*p3)])"),
(0.5,"TranslateY(percent=(-p1,p2),mode=mode_op[int(5*p3)])"),
(0.5,"Rotate((-p1,p2),mode=mode_op[int(5*p3)])"),
(0.5,"ShearX((-p1,p2),mode=mode_op[int(5*p3)])"),
(0.5,"ShearY((-p1,p2),mode=mode_op[int(5*p3)])"),
(0.5,"MultiplyBrightness((2*min([p1,p2]),2*max([p1,p2])))"),
(0.5,"AddToBrightness((-round(100*p1),round(100*p2)))"),
(0.5,"MultiplySaturation((2*min([p1,p2]),2*max([p1,p2])))"),
(0.5,"AddToSaturation((-round(100*p1),round(100*p2)))"),
(0.5,"GammaContrast((2*min([p1,p2]),2*max([p1,p2])))"),
(0.5,"LinearContrast((2*min([p1,p2]),2*max([p1,p2])))"),
(0.5,"HistogramEqualization()"),
(0.5,"GaussianBlur(sigma=3*p1)"),
(0.5,"AverageBlur(k=int(7*p1))"),
(0.5,"pillike.EnhanceContrast(factor=(0.5+p1,1.5-p2))"),
(0.5,"pillike.EnhanceBrightness(factor=(0.5+p1,1.5-p2))"),
(0.5,"pillike.EnhanceSharpness(factor=(0.5+p1,1.5-p2))"),
(0.5,"pillike.FilterSharpen()")
]

#number of inputs
n_inputs = len(input_functions)
#inputs defined as a list of integers
inputs = list(range(n_inputs))

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S') 


'''GA parameters'''
n_offspring = 4                   # number of offspring
n_genes_max = 5                   # maximum number of genes
#n_generations = 50               # number of generations
p_add = p_remove = p_mutate = 1/3 # actions probabilities
p_tf, p_pr = 0.5, 0.5


# defining the first individual size
n_genes_initial = np.random.randint(1,n_genes_max+1)
# creating the first individual
tf = np.random.randint(n_inputs, size=n_genes_initial)
param = np.random.random(size=(3,len(tf))).round(2)
parent = (tf,param)

if use_states:
    file = open('/home/states_seed'+str(seed)+'.pickle', 'rb')
    states = pickle.load(file)
    file.close()
    parent = states[0]
    timestamp = states[1]
    setState(states[2])
    EA.setState(states[3])
    DA.setState(states[4][0],states[4][1])

'''GA'''
best_fit_all = []
best_individual_all = []

for generation in range(start,stop):
    print("\n==================================================================")
    print("Generation: ", generation+1)
    print("==================================================================\n")
    print("Parent:",parent[0])
    
    # create offspring from one individual
    population = EA.create_offspring(parent, n_offspring, input_functions, p_add, p_remove, p_mutate, p_tf, p_pr,
                                     n_inputs, n_genes_max)
    print("Population:")
    for i in population:
        print(i[0])
        
    # measuring the fitness of each individual in the population
    [fitness, std_fitness] = EA.get_population_fitness(population,input_functions)
    print("\nFitness:",fitness)

    # save population statistics
    best_fit_all, best_individual_all = EA.export_stats(seed, timestamp, generation, population, 
                                        fitness, std_fitness, best_fit_all, best_individual_all)
    # selecting the best individual in the population as parent
    parent = (population[np.argmax(fitness)][0],population[np.argmax(fitness)][1])


toc = time.perf_counter()

if save_states:
    file = open('/home/states_seed'+str(seed)+'.pickle', 'wb')
    pickle.dump([parent, timestamp, getState(), EA.getState(), DA.getState()], file)
    file.close()
  
time_evol = toc-tic
time_hms = datetime.timedelta(seconds=time_evol)
time_file = open('/home/runtime_ES_tests.csv', 'a')
time_file.write('{};{}'.format(time_hms,time_evol)+'\n')
time_file.close()



    



