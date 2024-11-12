#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 17:02:58 2024

@author: shreyarajagopal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 12:58:00 2024

@author: shreyarajagopal
"""


#When using the functions to train the model, indices are numbered from 1 onwards
#Within actual functions, indices are numbered from 0 onwards as is standard for python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph


# Constants
NUM_CONTEXTS = 5
NUM_CUES = 5
NUM_ENGRAMS = 4  # Per type (positive, negative)
NUM_NEURONS = 4
FEATURES_DIM = 3  # For both contexts and cues
#NUM_FEATURES = NUM_CONTEXTS * FEATURES_DIM + NUM_CUES * FEATURES_DIM  # Total features for contexts and cues

class EnvironmentComponent:
    def __init__(self, num_items, dim):
        self.points = np.random.randint(1, 6, (num_items, dim))
        
class Engram:
    def __init__(self, num_engrams, num_neurons):
        self.neurons = np.zeros((num_engrams, num_neurons))

class ConnectionWeights:
    def __init__(self, source_labels, target_labels):
        self.weights = np.zeros((len(source_labels), len(target_labels)))
        self.source_labels = source_labels
        self.target_labels = target_labels

        
# Helper function to generate labels
def generate_labels(prefix, count, dim=FEATURES_DIM):
    labels = []
    for i in range(1, count + 1):
        for d in range(1, dim + 1):
            labels.append(f"{prefix}{i}D{d}")
    return labels

def generate_engram_labels(engram_type):
    """
    Generate labels for engram neurons, separating positive and negative engrams.
    :param engram_type: 'P' for positive, 'N' for negative
    :return: list of labels
    """
    labels = []
    for e in range(1, NUM_ENGRAMS + 1):
        for n in range(1, NUM_NEURONS + 1):
            labels.append(f"E{e}N{n}{engram_type}")  # P for positive, N for negative
    return labels


# Initialize components
contexts = EnvironmentComponent(NUM_CONTEXTS, FEATURES_DIM)
cues = EnvironmentComponent(NUM_CUES, FEATURES_DIM)
positive_engrams = Engram(NUM_ENGRAMS, NUM_NEURONS)
negative_engrams = Engram(NUM_ENGRAMS, NUM_NEURONS)

# Initialize labels
context_labels = generate_labels("C", NUM_CONTEXTS)
cue_labels = generate_labels("Q", NUM_CUES)
pos_engram_labels = generate_engram_labels('P')
neg_engram_labels = generate_engram_labels('N')

# Initialize connection weights
context_to_pos_engram_weights = ConnectionWeights(context_labels, pos_engram_labels)
context_to_neg_engram_weights = ConnectionWeights(context_labels, neg_engram_labels)
cue_to_pos_engram_weights = ConnectionWeights(cue_labels, pos_engram_labels)
cue_to_neg_engram_weights = ConnectionWeights(cue_labels, neg_engram_labels)


# Visualization functions
def visualize_points(points, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(title)
    plt.show()

# Visualization function adapted for labels
def visualize_weights(weights, title, xticklabels, yticklabels):
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(weights, annot=False, cmap='viridis', xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.show()


# Visualizing Contexts and Cues:
visualize_points(contexts.points, "Contexts in 3D Space")
visualize_points(cues.points, "Light Cues in 3D Space")

'''
#Visualizating connection weights:
    
#Cue-Engram Weights
visualize_weights(cue_to_neg_engram_weights.weights, "Cue to Negative Engram Weights", cue_to_neg_engram_weights.source_labels, cue_to_neg_engram_weights.target_labels)
visualize_weights(cue_to_pos_engram_weights.weights, "Cue to Positive Engram Weights", cue_to_pos_engram_weights.source_labels, cue_to_pos_engram_weights.target_labels)

#Context-Engram Weights
visualize_weights(context_to_neg_engram_weights.weights, "Context to Negative Engram Weights", context_to_neg_engram_weights.source_labels, context_to_neg_engram_weights.target_labels)
visualize_weights(context_to_pos_engram_weights.weights, "Context to Positive Engram Weights", context_to_pos_engram_weights.source_labels, context_to_pos_engram_weights.target_labels)

'''

def exp_decay(t, k=0.00921):
    """
    Calculate the decay factor based on a sigmoid curve for a given time t.
    
    Parameters:
    - t: Time in hours.
    - k: Steepness parameter of the sigmoid function.
    - x0: Midpoint of the sigmoid, where the function output is 0.5.
    
    Returns:
    - The decay factor based on the sigmoid curve at time t.
    """
    decay = 1 - np.exp(-k *t)
    return decay

def apply_decay_to_context_engram_weights(context_to_pos_engram_weights, context_to_neg_engram_weights, t=0.5, k=0.00921):
    """
    Apply decay to the weights of context to engram connections based on the time elapsed and a sigmoid curve.
    
    Parameters:
    - context_to_pos_engram_weights: ConnectionWeights instance for positive engrams.
    - context_to_neg_engram_weights: ConnectionWeights instance for negative engrams.
    - t: Time in hours since the start.
    - k: Steepness of the sigmoid curve for decay calculation.
    - x0: Midpoint of the sigmoid curve for decay calculation.
    """
    # Calculate the decay factor for the given time t
    decay_factor = exp_decay(t, k,)
    
    #CHANGE MARCH '24
    # Calculate the total decay to be applied, scaled by the time
    #total_decay = t * decay_factor
    
    # Apply the calculated decay to the weights, ensuring they do not drop below 0
    context_to_pos_engram_weights.weights = np.maximum(context_to_pos_engram_weights.weights -  decay_factor*context_to_pos_engram_weights.weights, 0)
    context_to_neg_engram_weights.weights = np.maximum(context_to_neg_engram_weights.weights - decay_factor*context_to_neg_engram_weights.weights, 0)
    
cue_neuron_indices = {} #For storing the indices of extinction engrams chosen for each cue
 

# Placeholder for the function that updates engram weights
def update_engram_weights(cue_index, context_index, cue_to_pos_engram_weights, context_to_pos_engram_weights, cue_to_neg_engram_weights, context_to_neg_engram_weights, aversive_present, appetitive_present, modify_cue_engram_weights, net_pos_sum, net_neg_sum, engram_index, upper_bound=1000):
    global cue_neuron_indices  # Refer to the global dictionary
    start_cue_dim = (cue_index - 1) * FEATURES_DIM
    end_cue_dim = start_cue_dim  + FEATURES_DIM
    cue_indices = range(start_cue_dim, end_cue_dim)
    
    start_context_dim =(context_index - 1) * FEATURES_DIM
    end_context_dim = start_context_dim  + FEATURES_DIM
    context_indices = range(start_context_dim, end_context_dim)

    # Handle the case when no US is present and no engram index is provided
    if engram_index is None:
        cue_index = cue_index-1
        # If this cue does not have stored neuron indices, select and store them #random 4 neurons selected as extinction engram (extinguishing negtaive or positive reward association)
        if cue_index not in cue_neuron_indices:
            cue_neuron_indices[cue_index] = np.random.choice(range(16), size=4, replace=False)
        selected_indices = cue_neuron_indices[cue_index] # once selected, same 4 neurons are always selected as extinction neurons for a specific cue
    else:
        # Calculate indices for the provided engram index as before
        neurons_per_engram = NUM_NEURONS
        start_index = (engram_index - 1) * neurons_per_engram
        end_index = start_index + neurons_per_engram
        selected_indices = range(start_index, end_index)

    update_context_indices = [(row, col) for row in context_indices for col in selected_indices]
    update_cue_indices = [(row, col) for row in cue_indices for col in selected_indices]


    # Update weights only for selected neuron indices
    if aversive_present and net_neg_sum < upper_bound:
        for row, col in update_context_indices:
            context_to_neg_engram_weights.weights[row, col] += 0.5
            #Ensure weight updation stops if net_neg_sum crosses upper_bound within the loop
            net_pos_sum_AV, net_neg_sum_AV = compute_weighted_sums(cue_index, context_index, cue_to_pos_engram_weights, context_to_pos_engram_weights, cue_to_neg_engram_weights, context_to_neg_engram_weights, sensitivity_scaling_vector)
            if net_neg_sum_AV > upper_bound:
                context_to_neg_engram_weights.weights[row, col] -= 0.5
        
        if modify_cue_engram_weights:
            for row, col in update_cue_indices:
                cue_to_neg_engram_weights.weights[row, col] += 0.7 
                #CHANGED March 2024 from +1
                   
                
    if appetitive_present and net_pos_sum < upper_bound:
        for row, col in update_context_indices:
            context_to_pos_engram_weights.weights[row, col] += 0.5
            #Ensure weight updation stops if net_pos_sum crosses upper_bound within the loop
            net_pos_sum_AP, net_neg_sum_AP = compute_weighted_sums(cue_index, context_index, cue_to_pos_engram_weights, context_to_pos_engram_weights, cue_to_neg_engram_weights, context_to_neg_engram_weights, sensitivity_scaling_vector)
            if net_pos_sum_AP> upper_bound:
                context_to_pos_engram_weights.weights[row, col] -= 0.5
        
        if modify_cue_engram_weights:
            for row, col in update_cue_indices:
                cue_to_pos_engram_weights.weights[row, col] += 0.7
                #Changed marcg 2024 from +1
                
        
       
            net_pos_sum = upper_bound

#sensitivity_scaling_vector = np.array([1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4])
#sensitivity_scaling_vector = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1])

sensitivity_scaling_vector=1

#Function for computing negative and positive engram sums scaled by different sensitivity levels to different USs:
def compute_weighted_sums(cue_index, context_index, cue_to_pos_engram_weights, context_to_pos_engram_weights, cue_to_neg_engram_weights, context_to_neg_engram_weights, sensitivity_scaling_vector):
   
    start_cue_dim = (cue_index - 1) * FEATURES_DIM
    end_cue_dim = start_cue_dim  + FEATURES_DIM
    cue_indices = range(start_cue_dim, end_cue_dim)
    
    start_context_dim =(context_index - 1) * FEATURES_DIM
    end_context_dim = start_context_dim  + FEATURES_DIM
    context_indices = range(start_context_dim, end_context_dim)

   # Extract the weights for the given cue and context
    cue_weights_pos = cue_to_pos_engram_weights.weights[cue_indices]
    context_weights_pos = context_to_pos_engram_weights.weights[context_indices]
    cue_weights_neg = cue_to_neg_engram_weights.weights[cue_indices]
    context_weights_neg = context_to_neg_engram_weights.weights[context_indices]
    
    # Compute the weighted sum for positive engrams
    net_pos_sum = np.sum(cue_weights_pos * sensitivity_scaling_vector) + np.sum(context_weights_pos * sensitivity_scaling_vector)
    
    # Compute the weighted sum for negative engrams
    net_neg_sum = np.sum(cue_weights_neg * sensitivity_scaling_vector) + np.sum(context_weights_neg * sensitivity_scaling_vector)
    #print(f"net negative sum = ({net_neg_sum})")
    
    return net_pos_sum, net_neg_sum   
    
# Placeholder for computing the final fear level
def compute_final_fear(net_pos_sum, net_neg_sum):
    # Step 2: Multiply net_neg_sum by -1
    net_neg_sum *= -1
    
    # Step 3: Compute net difference
    net_difference = net_pos_sum + net_neg_sum
    print(f"net_negative sum in this trial is = ({net_neg_sum})")
    
    # Step 4: Apply tanh function with scaling factor k
    k = 0.01
    output = np.tanh(k * net_difference)
    
    # Step 5: Determine the emotional response
    emotional_response = "aversive" if output < 0 else "appetitive"
    
    # Step 6: Display the final fear level
    print(f"Emotional Response at the end of this trial = ({emotional_response}, {output})")
    return output


# Function to execute the training session logic
def execute_training_trial(cue_index,context_index, aversive_present, appetitive_present, engram_index,time_between_trials =0.5):
 
    modify_cue_engram_weights = False
    
   
    # Apply decay to context-engram weights
    apply_decay_to_context_engram_weights(context_to_pos_engram_weights, context_to_neg_engram_weights,time_between_trials)
    
    #Compute Net Positive and Net Negative sums of BLA appetitive and aversive engrams (scaled by US Sensitivity)
    net_pos_sum, net_neg_sum = compute_weighted_sums(cue_index, context_index, cue_to_pos_engram_weights, context_to_pos_engram_weights, cue_to_neg_engram_weights, context_to_neg_engram_weights, sensitivity_scaling_vector)


    # Check if an unconditioned stimulus (US) is present
    US_present = aversive_present or appetitive_present
    
    if US_present:
        #print("Unconditioned Stimulus Present -- Cue Pathway weights will be updated")
        modify_cue_engram_weights = True
        
        # Function call to update engram weights based on the type of US present
        update_engram_weights(cue_index, context_index, cue_to_pos_engram_weights, context_to_pos_engram_weights, cue_to_neg_engram_weights, context_to_neg_engram_weights, aversive_present, appetitive_present, modify_cue_engram_weights, net_pos_sum, net_neg_sum, engram_index,upper_bound=1000,)
     
    elif not US_present and net_pos_sum != net_neg_sum:
        #print("No Unconditioned Stimulus Present,Conditioned State (Either Cue or Context or both are conditioned -- Only Context Pathway weights will be modified")
       
        if net_pos_sum < net_neg_sum:
            appetitive_present = True
            upper_bound = net_neg_sum
            #print("Fear Conditioned State")
        
        else: 
            aversive_present = True
            upper_bound = net_pos_sum
            #print("Appetitive Conditioned State")
        
        update_engram_weights(cue_index, context_index, cue_to_pos_engram_weights, context_to_pos_engram_weights, cue_to_neg_engram_weights, context_to_neg_engram_weights, aversive_present, appetitive_present, modify_cue_engram_weights, net_pos_sum, net_neg_sum, upper_bound = upper_bound, engram_index = None,)

    else:
        print("Netural State ")
        
    #Compute and display the final fear level
    #Calculate updated negative and positive sum
    net_pos_sum, net_neg_sum = compute_weighted_sums(cue_index, context_index, cue_to_pos_engram_weights, context_to_pos_engram_weights, cue_to_neg_engram_weights, context_to_neg_engram_weights, sensitivity_scaling_vector)
    #print("End of trial Net positive sum:", net_pos_sum)
    #print("End of trial Net negative sum:", net_neg_sum)
    
    final_fear = compute_final_fear(net_pos_sum,net_neg_sum)
    return final_fear
  
            
#TESTING
'''

#SET 1: Acquisition Context A, Extinction Context B, Testing Context A, B, C (ABA,ABB, ABC)

#Context-Based 
#Step 1: Fear Acquisition
trial_and_fear_values = []

final_fear_acq  = 0
trial_and_fear_values.append([1, final_fear_acq])

print("*****Cue 1 in Context 2 with US 4 acquisition****")
for i in range (0,15):
    print("Acquisition trial", i+2)
    print(context_to_neg_engram_weights.weights)
    final_fear_acq = execute_training_trial(1, 2, aversive_present = True, appetitive_present = False, engram_index = 4, time_between_trials = 0.05)
    trial_and_fear_values.append([i+2, final_fear_acq])

trial_and_fear_np_array = np.array(trial_and_fear_values)

#Visualize weights
#Cue-Engram Weights
#visualize_weights(cue_to_neg_engram_weights.weights.T, "Cue to Negative Engram Weights", cue_to_neg_engram_weights.source_labels, cue_to_neg_engram_weights.target_labels)
#visualize_weights(cue_to_pos_engram_weights.weights, "Cue to Positive Engram Weights", cue_to_pos_engram_weights.source_labels, cue_to_pos_engram_weights.target_labels)

#Context-Engram Weights
#visualize_weights(context_to_neg_engram_weights.weights.T, "Context to Negative Engram Weights", context_to_neg_engram_weights.source_labels, context_to_neg_engram_weights.target_labels)
#visualize_weights(context_to_pos_engram_weights.weights, "Context to Positive Engram Weights", context_to_pos_engram_weights.source_labels, context_to_pos_engram_weights.target_labels)

#Step 2: Fear Extinction 
trial_and_fear_values_ext = []

print("*****Extinguishing in Different Context: Cue 1 in Context 3 with no Us ***")  
print("Context Pathway Based Extinction - extinction with no reward") 

#first extinction trial happens 0.05 hours after acquisition
print("Extinction trial 1")
final_fear_ext = execute_training_trial(1, 3, aversive_present = False, appetitive_present = False, engram_index = None,time_between_trials = 180)
trial_and_fear_values_ext.append([1, final_fear_ext])

for i in range (0,40):
    print("Extinction trial", i+2)
    final_fear_ext = execute_training_trial(1, 3, aversive_present = False, appetitive_present = False, engram_index = None,time_between_trials = 0.05)
    trial_and_fear_values_ext.append([i+2, final_fear_ext])
    
trial_and_fear_ext_np_array = np.array(trial_and_fear_values_ext)


    #Visualize weights after each update:
    #Cue-Engram Weights
    #visualize_weights(cue_to_neg_engram_weights.weights.T, "Cue to Negative Engram Weights", cue_to_neg_engram_weights.source_labels, cue_to_neg_engram_weights.target_labels)
    #visualize_weights(cue_to_pos_engram_weights.weights, "Cue to Positive Engram Weights", cue_to_pos_engram_weights.source_labels, cue_to_pos_engram_weights.target_labels)

    #Context-Engram Weights
    #visualize_weights(context_to_pos_engram_weights.weights.T, "Context to Positive Engram Weights", context_to_neg_engram_weights.source_labels, context_to_neg_engram_weights.target_labels)
    #visualize_weights(context_to_pos_engram_weights.weights, "Context to Positive Engram Weights", context_to_pos_engram_weights.source_labels, context_to_pos_engram_weights.target_labels)
    #visualize_weights(context_to_neg_engram_weights.weights.T, "Context to Negative Engram Weights", context_to_neg_engram_weights.source_labels, context_to_neg_engram_weights.target_labels)


import matplotlib.pyplot as plt
import numpy as np

# Your provided code snippets would result in trial_and_fear_np_array and trial_and_fear_ext_np_array being defined here

# Example arrays for demonstration purposes
# Assuming trial_and_fear_np_array and trial_and_fear_ext_np_array are your data arrays
# trial_and_fear_np_array = np.array(trial_and_fear_values)
# trial_and_fear_ext_np_array = np.array(trial_and_fear_values_ext)


#Return of Fear Effects

print("****Renewal***")

print("****Renewal in Novel Context(neither Acquisition nor xtinction Context) context 4 (Immediately --> 3 minutes later)****")
final_fear_renewal_novel = execute_training_trial(1, 4, aversive_present = False, appetitive_present = False, engram_index = None, time_between_trials = 0.08)

print("****Renewal in Acquisition Context (Immediately --> 3 minutes later)****")
final_fear_renewal_acq = execute_training_trial(1, 2, aversive_present = False, appetitive_present = False, engram_index = None, time_between_trials = 0.08)

print("****Renewal in Extinction Context (Immediately --> 3 minutes later)****")
final_fear_renewal_ext = execute_training_trial(1, 3, aversive_present = False, appetitive_present = False, engram_index = None, time_between_trials = 0.08)


#print("****Renewal in Novel Context(neither Acquisition nor xtinction Context) context 4 (Immediately --> 3 minutes later)****")
#final_fear_renewal_novel = execute_training_trial(1, 4, aversive_present = False, appetitive_present = False, engram_index = None, time_between_trials = 0.05)


# Assuming trial_and_fear_np_array and trial_and_fear_ext_np_array are defined

# Flip the fear levels such that max fear is now +1 instead of -1
trial_and_fear_np_array[:, 1] = -1 * trial_and_fear_np_array[:, 1]
trial_and_fear_ext_np_array[:, 1] = -1 * trial_and_fear_ext_np_array[:, 1]

# Calculate the offset based on the last trial number of the acquisition phase
offset = trial_and_fear_np_array[:, 0].max() - trial_and_fear_ext_np_array[:, 0].min() + 1

# Apply the offset to the trial numbers for the extinction trials
trial_and_fear_ext_np_array[:, 0] += offset

# Plotting
plt.figure(figsize=(10, 6))

# Plot acquisition trials
plt.plot(trial_and_fear_np_array[:, 0], trial_and_fear_np_array[:, 1], label='Context A: Fear Acquisition', marker='o')

# Plot extinction trials with adjusted trial numbers
plt.plot(trial_and_fear_ext_np_array[:, 0], trial_and_fear_ext_np_array[:, 1], label='Context B: Fear Extinction', marker='x')

# Adding titles and labels
plt.title('Fear Levels Across Trials')
plt.xlabel('Trial Number')
plt.ylabel('Fear Level')
plt.legend()

# Show the plot
plt.show()

# Renewal points data, assuming final_fear_renewal_acq, final_fear_renewal_novel, and final_fear_renewal_ext are defined
contexts = ['ABA: Acquisition Context', 'ABC: Novel Context', 'ABB: Extinction Context']
fear_levels = [-1 * final_fear_renewal_acq, -1 * final_fear_renewal_novel, -1 * final_fear_renewal_ext]

# Plotting the bar graph
plt.figure(figsize=(8, 6))
bar_positions = range(len(contexts))  # Positions for the bars on the x-axis
plt.bar(bar_positions, fear_levels, color=['red', 'green', 'orange'])

# Add titles and labels
plt.title('Fear Renewal Levels in Different Contexts')
plt.xticks(bar_positions, contexts)  # Set the x-ticks to match the contexts
plt.ylabel('Fear Level')

# Show the plot
plt.show()
'''

'''
#SET 2: Acquisition Context= Extinction Context = A, Testing Contexts A and B (AAB, AAA)

#Context-Based 
#Step 1: Fear Acquisition
trial_and_fear_values = []

final_fear_acq  = 0
trial_and_fear_values.append([1, final_fear_acq])

print("*****Cue 1 in Context 2 with US 4 acquisition****")
for i in range (0,15):
    print("Acquisition trial", i+2)
    final_fear_acq = execute_training_trial(1, 2, aversive_present = True, appetitive_present = False, engram_index = 4, time_between_trials = 0.05)
    trial_and_fear_values.append([i+2, final_fear_acq])

trial_and_fear_np_array = np.array(trial_and_fear_values)

#Visualize weights
#Cue-Engram Weights
#visualize_weights(cue_to_neg_engram_weights.weights.T, "Cue to Negative Engram Weights", cue_to_neg_engram_weights.source_labels, cue_to_neg_engram_weights.target_labels)
#visualize_weights(cue_to_pos_engram_weights.weights, "Cue to Positive Engram Weights", cue_to_pos_engram_weights.source_labels, cue_to_pos_engram_weights.target_labels)

#Context-Engram Weights
#visualize_weights(context_to_neg_engram_weights.weights.T, "Context to Negative Engram Weights", context_to_neg_engram_weights.source_labels, context_to_neg_engram_weights.target_labels)
#visualize_weights(context_to_pos_engram_weights.weights, "Context to Positive Engram Weights", context_to_pos_engram_weights.source_labels, context_to_pos_engram_weights.target_labels)

#Step 2: Fear Extinction 
trial_and_fear_values_ext = []

print("*****Extinguishing in Different Context: Cue 1 in Context 2 with no Us ***")  
print("Context Pathway Based Extinction - extinction with no reward") 

#first extinction trial happens 0.5 hours after acquisition
print("Extinction trial 1")
final_fear_ext = execute_training_trial(1, 2, aversive_present = False, appetitive_present = False, engram_index = None,time_between_trials = 0.05)
trial_and_fear_values_ext.append([1, final_fear_ext])

for i in range (0,40):
    print("Extinction trial", i+2)
    final_fear_ext = execute_training_trial(1, 2, aversive_present = False, appetitive_present = False, engram_index = None,time_between_trials = 0.05)
    trial_and_fear_values_ext.append([i+2, final_fear_ext])
    
trial_and_fear_ext_np_array = np.array(trial_and_fear_values_ext)


    #Visualize weights after each update:
    #Cue-Engram Weights
    #visualize_weights(cue_to_neg_engram_weights.weights.T, "Cue to Negative Engram Weights", cue_to_neg_engram_weights.source_labels, cue_to_neg_engram_weights.target_labels)
    #visualize_weights(cue_to_pos_engram_weights.weights, "Cue to Positive Engram Weights", cue_to_pos_engram_weights.source_labels, cue_to_pos_engram_weights.target_labels)

    #Context-Engram Weights
    #visualize_weights(context_to_pos_engram_weights.weights.T, "Context to Positive Engram Weights", context_to_neg_engram_weights.source_labels, context_to_neg_engram_weights.target_labels)
    #visualize_weights(context_to_pos_engram_weights.weights, "Context to Positive Engram Weights", context_to_pos_engram_weights.source_labels, context_to_pos_engram_weights.target_labels)
    #visualize_weights(context_to_neg_engram_weights.weights.T, "Context to Negative Engram Weights", context_to_neg_engram_weights.source_labels, context_to_neg_engram_weights.target_labels)


import matplotlib.pyplot as plt
import numpy as np

# Your provided code snippets would result in trial_and_fear_np_array and trial_and_fear_ext_np_array being defined here

# Example arrays for demonstration purposes
# Assuming trial_and_fear_np_array and trial_and_fear_ext_np_array are your data arrays
# trial_and_fear_np_array = np.array(trial_and_fear_values)
# trial_and_fear_ext_np_array = np.array(trial_and_fear_values_ext)


#Return of Fear Effects

print("****Renewal***")

print("****Renewal in Novel Context(neither Acquisition nor xtinction Context) context 4 (Immediately --> 3 minutes later)****")
final_fear_renewal_novel = execute_training_trial(1, 4, aversive_present = False, appetitive_present = False, engram_index = None, time_between_trials = 0.08)

print("****Renewal in Acquisition Context (Immediately --> 3 minutes later)****")
final_fear_renewal_acq = execute_training_trial(1, 2, aversive_present = False, appetitive_present = False, engram_index = None, time_between_trials = 0.08)


import matplotlib.pyplot as plt
import numpy as np

# Assuming trial_and_fear_np_array and trial_and_fear_ext_np_array are defined, along with final_fear_renewal_acq and final_fear_renewal_novel

# Flip the fear levels such that max fear is now +1 instead of -1
trial_and_fear_np_array[:, 1] = -1 * trial_and_fear_np_array[:, 1]
trial_and_fear_ext_np_array[:, 1] = -1 * trial_and_fear_ext_np_array[:, 1]

# Calculate the offset for the extinction trials to start after acquisition trials
offset = trial_and_fear_np_array[:, 0].max() - trial_and_fear_ext_np_array[:, 0].min() + 1
trial_and_fear_ext_np_array[:, 0] += offset

# Plotting
plt.figure(figsize=(10, 6))

# Plot acquisition trials
plt.plot(trial_and_fear_np_array[:, 0], trial_and_fear_np_array[:, 1], label='Context A: Fear Acquisition', marker='o')

# Plot extinction trials with the adjusted start
plt.plot(trial_and_fear_ext_np_array[:, 0], trial_and_fear_ext_np_array[:, 1], label='Context A: Fear Extinction', marker='x')

# Adding titles and labels
plt.title('Fear Levels Across Trials - Acquisition and Extinction Contexts are the Same')
plt.xlabel('Trial Number')
plt.ylabel('Fear Level')
plt.legend()

# Show the plot
plt.show()

# Adjusting the order of renewal points data
contexts = ['Novel Context(AAB)', 'Acquisition Context(AAA)']  # Novel context comes first
fear_levels = [-1 * final_fear_renewal_novel, -1 * final_fear_renewal_acq]  # Corresponding fear levels order adjusted

# Plotting the bar graph with adjusted order
plt.figure(figsize=(8, 6))
bar_positions = range(len(contexts))  # Positions for the bars on the x-axis
plt.bar(bar_positions, fear_levels, color=['orange', 'red'])  # Colors order adjusted to match

# Add titles and labels
plt.title('Fear Renewal Levels in Different Contexts')
plt.xticks(bar_positions, contexts)  # Set the x-ticks to match the new order of contexts
plt.ylabel('Fear Level')

# Show the plot
plt.show()

'''

'''

#SET 3: Spontaneous Recovery 

#Context-Based 
#Step 1: Fear Acquisition
trial_and_fear_values = []

final_fear_acq  = 0
trial_and_fear_values.append([1, final_fear_acq])

print("*****Cue 1 in Context 2 with US 4 acquisition****")
for i in range (0,15):
    print("Acquisition trial", i+2)
    final_fear_acq = execute_training_trial(1, 2, aversive_present = True, appetitive_present = False, engram_index = 4, time_between_trials = 0.05)
    trial_and_fear_values.append([i+2, final_fear_acq])

trial_and_fear_np_array = np.array(trial_and_fear_values)

#Visualize weights
#Cue-Engram Weights
#visualize_weights(cue_to_neg_engram_weights.weights.T, "Cue to Negative Engram Weights", cue_to_neg_engram_weights.source_labels, cue_to_neg_engram_weights.target_labels)
#visualize_weights(cue_to_pos_engram_weights.weights, "Cue to Positive Engram Weights", cue_to_pos_engram_weights.source_labels, cue_to_pos_engram_weights.target_labels)

#Context-Engram Weights
#visualize_weights(context_to_neg_engram_weights.weights.T, "Context to Negative Engram Weights", context_to_neg_engram_weights.source_labels, context_to_neg_engram_weights.target_labels)
#visualize_weights(context_to_pos_engram_weights.weights, "Context to Positive Engram Weights", context_to_pos_engram_weights.source_labels, context_to_pos_engram_weights.target_labels)

#Step 2: Fear Extinction 
trial_and_fear_values_ext = []

print("*****Extinguishing in Different Context: Cue 1 in Context 2 with no Us ***")  
print("Context Pathway Based Extinction - extinction with no reward") 

#first extinction trial happens 0.5 hours after acquisition
print("Extinction trial 1")
final_fear_ext = execute_training_trial(1, 2, aversive_present = False, appetitive_present = False, engram_index = None,time_between_trials = 0.05)
trial_and_fear_values_ext.append([1, final_fear_ext])

for i in range (0,40):
    print("Extinction trial", i+2)
    final_fear_ext = execute_training_trial(1, 2, aversive_present = False, appetitive_present = False, engram_index = None,time_between_trials = 0.05)
    trial_and_fear_values_ext.append([i+2, final_fear_ext])
    
trial_and_fear_ext_np_array = np.array(trial_and_fear_values_ext)


#Step 3: 

print("SR!!")

trial_and_fear_values_SR = []
list = [0,1,2,4,6,10,14,21]
l = 1
for i in list:
    print("Spontaneous Recovery Trial", l)
    final_fear_sr = execute_training_trial(1, 2, aversive_present = False, appetitive_present = False, engram_index = None,time_between_trials = i*24)
    trial_and_fear_values_SR.append([l, final_fear_sr])
    l = l+1
    
trial_and_fear_SR_np_array = np.array(trial_and_fear_values_SR)

#Plot
# Example list for x-axis
x_axis_values = [0, 1, 2, 4, 6, 10, 14, 21]

# Assuming trial_and_fear_SR_np_array's first column represents these specific trials or times directly
# And its second column represents fear levels
fear_levels = trial_and_fear_SR_np_array[:, 1]
fear_levels = fear_levels * -1

# Generate equally spaced indices for the x-axis positions of the bars
indices = np.arange(len(x_axis_values))

# Plotting as a bar chart
plt.figure(figsize=(10, 6))

# Plot the data as a bar chart using the indices for equal spacing
plt.bar(indices, fear_levels, color='b', label='Fear Levels over Time')

# Set the x-axis ticks to correspond to the original x-axis values, using the indices for positioning
plt.xticks(indices, x_axis_values)

# Adding titles and labels
plt.title('Fear Levels Across Spontaneous Recovery Trials')
plt.xlabel('Days after Extinction')
plt.ylabel('Fear Level')

plt.legend()
plt.show()

#final_fear_sr = execute_training_trial(1, 2, aversive_present = False, appetitive_present = False, engram_index = None,time_between_trials =500)




'''






#Option 2: CounterConditioning

#Step 1: Fear Acquisition
trial_and_fear_values = []

final_fear_acq  = 0
trial_and_fear_values.append([1, final_fear_acq])

print("*****Cue 1 in Context 2 with US 4 acquisition****")
for i in range (0,15):
    print("Acquisition trial", i+2)
    final_fear_acq = execute_training_trial(1, 2, aversive_present = True, appetitive_present = False, engram_index = 4, time_between_trials = 0.05)
    trial_and_fear_values.append([i+2, final_fear_acq])

trial_and_fear_np_array = np.array(trial_and_fear_values)

#Visualize weights
#Cue-Engram Weights
#visualize_weights(cue_to_neg_engram_weights.weights.T, "Cue to Negative Engram Weights", cue_to_neg_engram_weights.source_labels, cue_to_neg_engram_weights.target_labels)
#visualize_weights(cue_to_pos_engram_weights.weights, "Cue to Positive Engram Weights", cue_to_pos_engram_weights.source_labels, cue_to_pos_engram_weights.target_labels)

#Context-Engram Weights
#visualize_weights(context_to_neg_engram_weights.weights.T, "Context to Negative Engram Weights", context_to_neg_engram_weights.source_labels, context_to_neg_engram_weights.target_labels)
#visualize_weights(context_to_pos_engram_weights.weights, "Context to Positive Engram Weights", context_to_pos_engram_weights.source_labels, context_to_pos_engram_weights.target_labels)

#Step 2: Counterconditioning in Novel Context
trial_and_fear_values_cc = []

print("*****Counterconditioningin Different Context: Cue 1 in Context 3 with positive Us ***")  
print("Context Pathway Based Extinction - extinction with no reward") 

#first extinction trial happens 0.5 hours after acquisition
print("Extinction trial 1")
final_fear_ext = execute_training_trial(1, 3, aversive_present = False, appetitive_present = True, engram_index = 3,time_between_trials = 0.05)
trial_and_fear_values_cc.append([1, final_fear_ext])

for i in range (0,8):
    print("Extinction trial", i+2)
    final_fear_cc = execute_training_trial(1, 3, aversive_present = False, appetitive_present =True, engram_index = None,time_between_trials = 0.05)
    trial_and_fear_values_cc.append([i+2, final_fear_cc])
    
trial_and_fear_cc_np_array = np.array(trial_and_fear_values_cc
                                      )
print("****Testing in Acquisition Context after counterconditioning (0.5 hours)***")
final_fear_cc_acq = execute_training_trial(1, 2, aversive_present = False, appetitive_present = False, engram_index = None)

print("Testing in counterconditioning context after extinguishing with positive cue in acquisition context")
final_fear_cc_cc = execute_training_trial(1, 3, aversive_present = False, appetitive_present = False, engram_index = None)

print("Testing in novel context after ecounterconditioning")
final_fear_cc_nov  = execute_training_trial(1, 4, aversive_present = False, appetitive_present = False, engram_index = None)

print("Testing in Counterconditioning Context after 21 days (500 hours))")
final_fear_cc_cc_SR = execute_training_trial(1, 3, aversive_present = False, appetitive_present = False, engram_index = None, time_between_trials=500)


# Plotting
# Flip the fear levels such that max fear is now +1 instead of -1
trial_and_fear_np_array[:, 1] = -1 * trial_and_fear_np_array[:, 1]
trial_and_fear_cc_np_array[:, 1] = -1 * trial_and_fear_cc_np_array[:, 1]

# Calculate the offset for the counterconditioning trials to ensure they follow acquisition trials
offset = trial_and_fear_np_array[:, 0].max() - trial_and_fear_cc_np_array[:, 0].min() + 1
trial_and_fear_cc_np_array[:, 0] += offset

# Plotting
plt.figure(figsize=(10, 6))

# Plot acquisition trials
plt.plot(trial_and_fear_np_array[:, 0], trial_and_fear_np_array[:, 1], label='Context A: Fear Acquisition', marker='o')

# Plot counterconditioning trials with the adjusted start
plt.plot(trial_and_fear_cc_np_array[:, 0], trial_and_fear_cc_np_array[:, 1], label='Context B: Counterconditioning ', marker='x',color='purple')

# Adding titles and labels
plt.title('Fear Levels Across Trials')
plt.xlabel('Trial Number')
plt.ylabel('Fear Level')
plt.legend()

# Show the plot
plt.show()

# Data for renewal points
contexts = ['Acquisition Context (ABA)', 'Counterconditioning Context(ABB)', 'Novel Context(ABC)', 'Spontaneous Recovery in CC Context']
fear_levels = [-1 * final_fear_cc_acq, -1 * final_fear_cc_cc, -1 * final_fear_cc_nov, -1 * final_fear_cc_cc_SR]

# Plotting the bar graph
plt.figure(figsize=(10, 6))
bar_positions = range(len(contexts))  # Positions for the bars on the x-axis
colors = ['red', 'green', 'orange', 'pink']  # Colors corresponding to the original scatter plot
plt.bar(bar_positions, fear_levels, color=colors)

# Add titles and labels
plt.title('Fear Renewal Levels in Different Contexts')
plt.xticks(bar_positions, contexts, rotation=45)  # Set the x-ticks to match the contexts, with a rotation for better readability
plt.ylabel('Fear Level')

# Show the plot
plt.show()


#Comparing SR in Counterconditioning and Extinction
# Bar graph values and settings
values = [0.85, 0.4]
bar_labels = ['Extinction', 'Counterconditioning']
colors = ['blue', 'red']

# Plotting the bar graph
plt.figure(figsize=(8, 6))
bars = plt.bar(bar_labels, values, color=colors)

# Adding titles and labels with a second line for the y-axis label
plt.title('Comparing Spontaneous Recovery')
plt.xlabel('')
plt.ylabel('Fear Level\nAfter 21 days')

# Show the plot
plt.show()





#Decay +ve weights also, but more slowly than context weights!!
#Different time courses of decay for each context's weights 

    
