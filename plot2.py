import matplotlib.pyplot as plt
import numpy as np
import pickle

# read result from pickle file
def read_result(file_path):
    with open(file_path, 'rb') as file:
        result = pickle.load(file)
    return result

def plot_histograms_with_stats(model1_versions, model2_versions):
    metrics = ['test_error', 'test_accuracy', 'test_time']
    titles = ['Test Error', 'Test Accuracy', 'Test Time']
    colors = ['blue', 'orange']
    alphas = [0.3, 0.5, 0.7]
    labels = ['Version 1', 'Version 2', 'Version 3']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, metric in enumerate(metrics):
        for version_index in range(3):
            model1 = model1_versions[version_index]
            model2 = model2_versions[version_index]
            
            # Determine common bins for both histograms
            all_data = np.concatenate((model1[metric], model2[metric]))
            counts, bins = np.histogram(all_data, bins=20)
            
            # Plot histograms
            axes[i].hist(model1[metric], bins=bins, alpha=alphas[version_index], label=f'Model 1 {labels[version_index]}', color=colors[0], histtype='stepfilled')
            axes[i].hist(model2[metric], bins=bins, alpha=alphas[version_index], label=f'Model 2 {labels[version_index]}', color=colors[1], histtype='stepfilled')
        
        # Set titles and labels
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    # Save the plot
    plt.savefig('performance_comparison_versions.png')

if __name__ == '__main__':
    batch_size = [1,8,32,64]
    model1_versions = [read_result(f'model1_performance_v{i}.pkl') for i in range(1, 4)]
    model2_versions = [read_result(f'model2_performance_v{i}.pkl') for i in range(1, 4)]

    plot_histograms_with_stats(model1_versions, model2_versions)
