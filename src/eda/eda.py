import os
import matplotlib.pyplot as plt

def do_eda():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    suspicious_dir = os.path.join(data_dir, 'suspicious')
    normal_dir = os.path.join(data_dir, 'normal')


    report_path = os.path.join(os.path.dirname(__file__), 'eda_report.txt')
    plot_path = os.path.join(os.path.dirname(__file__), 'eda_distribution.png')
    
    # gather counts
    counts = {}
    total_suspicious = 0
    total_normal = 0

    if os.path.exists(suspicious_dir):
        for root, dirs, files in os.walk(suspicious_dir):
            if root != suspicious_dir:
                category = os.path.basename(root)
                num_files = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if num_files > 0:
                    counts[f'Suspicious: {category}'] = num_files
                    total_suspicious += num_files

    if os.path.exists(normal_dir):
        for root, dirs, files in os.walk(normal_dir):
            if root != normal_dir:
                category = os.path.basename(root)
                num_files = len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if num_files > 0:
                    counts[f'Normal: {category}'] = num_files
                    total_normal += num_files
                    
    with open(report_path, 'w') as f:
        f.write("Exploratory Data Analysis Report\n")
        f.write("=" * 32 + "\n\n")
        f.write(f"Total Suspicious Images: {total_suspicious}\n")
        f.write(f"Total Normal Images: {total_normal}\n")
        f.write(f"Total Images Overall: {total_suspicious + total_normal}\n\n")
        f.write("Breakdown by Category:\n")
        for k, v in counts.items():
            f.write(f"  - {k}: {v} images\n")

    print(f"Report saved to {report_path}")

    # plot
    if counts:
        plt.figure(figsize=(12, 8))
        names = list(counts.keys())
        values = list(counts.values())
        
        # Color coding: red for suspicious, green for normal
        colors = ['red' if 'Suspicious' in n else 'green' for n in names]
        
        # Clean names for x-axis
        x_names = [n.split(': ')[1] for n in names]
        
        plt.bar(x_names, values, color=colors)
        plt.xticks(rotation=90)
        plt.ylabel('Number of Images')
        plt.title('Dataset Distribution by Subcategory')
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

if __name__ == '__main__':
    do_eda()
