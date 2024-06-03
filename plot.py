# Run this file if you need to generate comparision graph with ACE and marepo

import matplotlib.pyplot as plt
import numpy as np
import pickle

# read result from pickle file
def read_result(file_path):
    with open(file_path, 'rb') as file:
        result = pickle.load(file)
    return result

def plot_histograms_with_stats(model1, model2):
    metrics = ['rotation_error', 'translation_error', 'avg_processing_time']
    titles = ['Rotation Error(degree)', 'Translation Error(cm)', 'Avg Processing Time(s)']
    colors = ['blue', 'orange']
    print(len(model1['rotation_error']),len(model2['rotation_error']))
    print(len(model1['translation_error']),len(model2['translation_error']))
    print(len(model1['avg_processing_time']),len(model2['avg_processing_time']))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        all_data = np.concatenate((model1[metric], model2[metric]))
        counts, bins = np.histogram(all_data, bins=20)
        
        # Plot histograms
        axes[i].hist(model1[metric], bins=bins, alpha=0.5, label='ACE', color=colors[0])
        axes[i].hist(model2[metric], bins=bins, alpha=0.5, label='MAREPO(finetune 600 eps)', color=colors[1])
        
        # Calculate and plot mean and median for Model 1
        mean1 = np.mean(model1[metric])
        median1 = np.median(model1[metric])
        axes[i].axvline(mean1, color=colors[0], linestyle='dashed', linewidth=2, label='ACE Mean')
        axes[i].axvline(median1, color=colors[0], linestyle='solid', linewidth=2, label='ACE Median')
        
        # Calculate and plot mean and median for Model 2
        mean2 = np.mean(model2[metric])
        median2 = np.median(model2[metric])
        axes[i].axvline(mean2, color=colors[1], linestyle='dashed', linewidth=2, label='MAREPO Mean')
        axes[i].axvline(median2, color=colors[1], linestyle='solid', linewidth=2, label='MAREPO Median')
        
        # Set titles and labels
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    # Save the plot
    plt.savefig('performance_comparison.png')

if __name__ == '__main__':
    model1_performance = read_result('/workspace/project/marepo/logs/pretrain/ace_models/7Scenes/result_7scenes_testBL_.pkl')
    model2_performance = read_result('/workspace/project/marepo/logs/marepo_7scenes_testBL_240405/result_7scenes_testBL_.pkl')
    plot_histograms_with_stats(model1_performance, model2_performance)