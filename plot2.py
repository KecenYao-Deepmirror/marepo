import matplotlib.pyplot as plt
import numpy as np
import pickle

# read result from pickle file
def read_result(file_path):
    with open(file_path, 'rb') as file:
        result = pickle.load(file)
    return result

def plot_histograms_with_stats(model1_versions, model2_versions):
    metrics = ['avg_processing_time']
    titles = ['avg_processing_time']
    colors = ['blue', 'orange']
    alphas = [0.4, 0.6, 0.8, 0.9]
    labels = ['Batch Size 1', 'Batch Size 8', 'Batch Size 32','Batch Size 64']
    
    fig, axes = plt.subplots(1, 1, figsize=(6, 5))
    
    for i, metric in enumerate(metrics):
        for version_index in range(4):
            fig, axes = plt.subplots(1, 1, figsize=(6, 5))
            model1 = model1_versions[version_index]
            model2 = model2_versions[version_index]
            
            # Determine common bins for both histograms
            all_data = np.concatenate((model1[metric], model2[metric]))
            counts, bins = np.histogram(all_data, bins=100)
            
            # Plot histograms
            axes.hist(model1[metric], bins=bins, alpha=alphas[version_index], label=f'ACE {labels[version_index]}', color=colors[0], histtype='stepfilled',density=True)
            axes.hist(model2[metric], bins=bins, alpha=alphas[version_index], label=f'MAREPO {labels[version_index]}', color=colors[1], histtype='stepfilled',density=True)

            mean = np.mean(model1[metric])
            median = np.median(model1[metric])
            axes.axvline(mean, color=colors[0], linestyle='dashed', linewidth=2)
            axes.axvline(median, color=colors[0], linestyle='solid', linewidth=2)

            mean = np.mean(model2[metric])
            median = np.median(model2[metric])
            axes.axvline(mean, color=colors[1], linestyle='dashed', linewidth=2)
            axes.axvline(median, color=colors[1], linestyle='solid', linewidth=2)
            print(len(model1[metric]),len(model2[metric]))
            print(len(model1[metric]),max(model1[metric]))
            print(len(model2[metric]),max(model2[metric]))
        
            # Set titles and labels
            axes.set_title(titles[i])
            axes.set_xlabel(metric.replace('_', ' ').title())
            axes.set_ylabel('Frequency')
            axes.legend()
        
            plt.tight_layout()
            plt.show()
            # Save the plot
            plt.savefig(f'performance_comparison_versions_b{labels[version_index]}.png')

if __name__ == '__main__':
    model1_versions = [read_result(f'/workspace/project/marepo/logs/pretrain/ace_models/7Scenes/result_7scenes_testBL_b{i}.pkl') for i in [1,8,32,64]]
    model2_versions = [read_result(f'/workspace/project/marepo/logs/marepo_7scenes_testBL_240405/result_7scenes_testBL_b{i}.pkl') for i in [1,8,32,64]]

    plot_histograms_with_stats(model1_versions, model2_versions)
