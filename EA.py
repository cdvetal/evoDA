""" Genetic Algorithm Functions """

import numpy as np
import pickle
import DA
import copy
import imgaug.augmenters as iaa
np.random.seed(42)

def get_population_fitness(population,input_functions):
    ''' Calculating the fitness value of each individual in the current population '''
    # the fitness function is the validation ROC AUC value
    [fitness, std_fitness] = DA.get_AUC_values(population,input_functions)
    return fitness, std_fitness

def validate_individual(individual,input_functions):
    functions = [input_functions[i] for i in individual[0]]
    parameters = individual[1]
    mode_op = ['constant','edge','symmetric','reflect','wrap']
    try:
        for i in range(len(functions)):
            p1,p2,p3 = parameters[0,i],parameters[1,i],parameters[2,i]
            iaa.Sometimes(functions[i][0], eval('iaa.'+functions[i][1]))
        return True
    except:
        return False

def create_offspring(parent, n_offspring, input_functions, p_add, p_remove, p_mutate, p_tf, p_pr, n_inputs, n_genes_max):
    ''' Create offspring by performing some actions on the parent '''
    offspring = [parent] 
    tf = parent[0]
    pr = parent[1]
    for oo in range(n_offspring):       
        operation = np.random.random()
        new,new_pr = [],[]
        if operation >= p_pr:
            done = False
            while done == False:
                action = np.random.random()
                if action <= p_add:
                    #add random gene
                    if len(tf) < n_genes_max:
                        idx = np.random.randint(len(tf)) # index where to add
                        value = np.random.randint(n_inputs) # gene value to add
                        new = np.insert(tf, idx, value)                        
                        new_pr = np.array([np.insert(pr[i],idx,round(np.random.random(),2)) for i in range(len(pr))])
                        done = True
                elif action > p_add and action <= p_add+p_remove:
                    #remove random 
                    if len(tf) > 1:
                        idx = np.random.randint(len(tf)) # index to remove
                        new = np.delete(tf, idx)
                        new_pr = np.array([np.delete(pr[i],idx) for i in range(len(pr))])
                        done = True
                else:
                    #mutate random gene
                    idx = np.random.randint(len(tf)) # index where to mutate
                    value = np.random.randint(n_inputs) # new gene value
                    new = np.copy(tf)
                    new[idx] = value
                    new_pr = pr
                    done = True
                if done and tuple(new) != tuple(tf) and validate_individual((new,new_pr),input_functions):
                    offspring.append((new,new_pr))
                else:
                    done = False
        else:
            done = False
            while done == False:
                #mutate random parameter of a random transformation function 
                new_pr = copy.deepcopy(pr)
                idx = np.random.randint(len(tf)) # transformation function where to mutate
                value = round(np.random.random(),2) # new parameter value
                new = new_pr[:,idx]
                idx_pr = np.random.randint(3) #parameter index
                new[idx_pr] = value
                new_pr[:,idx] = new
                if validate_individual((tf,new_pr),input_functions):
                    offspring.append((tf,new_pr))
                    done = True
    return offspring

def export_stats(seed, timestamp, generation, population, fitness, std_fitness, best_fit_all, best_individual_all):
    ''' Save population statistics '''
    # simple exporter per generation...
    with open('/home/TestR4P_100G_Seed{0}.csv'.format(seed), 'a') as stats_f:
        
        save_path = '/home/TestR4P_100G_Seed{0}.pickle'.format(seed)
        
        if generation == 0:
            stats_f.write('Generation;Population TF;Population PR;Fitness;Std;Avg;Min;Best Fitness;Std;Best Individual TF;Best Individual PR\n')
            gen, pop, fit, std_fit = [],[],[],[]
        else:
            file = open(save_path, 'rb')
            [gen, pop, fit, std_fit] = pickle.load(file)
            file.close()
            
    
        # population information
        popavg = np.mean(fitness)
        popmin = np.min(fitness)
        
        # the best result in the current generation
        best_fit = np.max(fitness)
        std_best_fit = std_fitness[np.argmax(fitness)]
        best_individual = population[np.argmax(fitness)]
        print("\nBest generation result: ", best_fit)
        print("Best generation individual:\n", best_individual[0],'\n',best_individual[1])
        best_fit_all.append(best_fit)
        best_individual_all.append(best_individual)
    
        
        file = open(save_path, 'wb')
        gen.append(generation)
        pop.append(population)
        fit.append(fitness)
        std_fit.append(std_fitness)
        pickle.dump([gen, pop, fit, std_fit], file)
        file.close()
        del gen, pop, fit, std_fit
        
        stats_f.write('{};{};{};{};{};{};{};{};{};{};{}'.format(generation, str([p[0].tolist() for p in population]),
                      [p[1].tolist() for p in population], fitness*100, std_fitness*100, popavg*100, popmin*100, best_fit*100,
                      std_best_fit*100, str(best_individual[0]).replace(',',' '),("\""+str(best_individual[1])+"\"").replace(',','\n')).replace('.',',').replace('\n',' ') +'\n')
    return best_fit_all, best_individual_all

def getState():  
    return np.random.get_state()

def setState(state_np):
    np.random.set_state(state_np)
    