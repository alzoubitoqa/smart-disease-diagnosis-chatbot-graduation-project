# fix_figure_4_2.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Set correct paths
# BASE_DIR هو مجلد المشروع الرئيسي (gp_2)
BASE_DIR = r"C:\Users\User\OneDrive\Desktop\gp_2"
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "artifacts", "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

print(f"📁 Data directory: {DATA_DIR}")
print(f"📁 Figures directory: {FIGURES_DIR}")

# Check if files exist
dataset_path = os.path.join(DATA_DIR, "dataset.csv")
severity_path = os.path.join(DATA_DIR, "Symptom-severity.csv")

if not os.path.exists(dataset_path):
    print(f"❌ File not found: {dataset_path}")
    print("   Please check the path")
    exit(1)

if not os.path.exists(severity_path):
    print(f"❌ File not found: {severity_path}")
    exit(1)

# Load data
print("\n📂 Loading data...")
dataset = pd.read_csv(dataset_path)
severity = pd.read_csv(severity_path)

# Get symptom columns
symptom_cols = [c for c in dataset.columns if c.startswith("Symptom_")]

# Create severity dictionary
severity_dict = dict(zip(severity['Symptom'], severity['weight']))

# Extract severity values
print("\n📊 Extracting severity values...")
severity_values = []
for row in dataset[symptom_cols].values:
    for s in row:
        if pd.notna(s) and s != "<pad>" and s != "" and s != " ":
            sev = severity_dict.get(s, 1)
            severity_values.append(sev)

# Count severity levels
severity_counts = Counter(severity_values)
severity_levels = sorted(severity_counts.keys())
counts = [severity_counts[lev] for lev in severity_levels]

# Print data for verification
print(f"\n📈 Severity Levels: {severity_levels}")
print(f"📈 Counts: {counts}")
print(f"📈 Total severity occurrences: {sum(counts)}")

# Create figure
print("\n🎨 Creating figure...")
fig, ax = plt.subplots(figsize=(10, 6))

# Colors for each severity level
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

bars = ax.bar(severity_levels, counts, color=colors[:len(severity_levels)], 
              edgecolor='black', linewidth=1.5)

ax.set_xlabel('Severity Level', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Figure 4.2: Distribution of Symptom Severity Levels', fontsize=14, fontweight='bold')
ax.set_xticks(severity_levels)

# Add value labels on bars
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
            str(count), ha='center', va='bottom', fontsize=10)

plt.tight_layout()

# Save figure
png_path = os.path.join(FIGURES_DIR, "Figure_4.2_Severity_Distribution.png")
pdf_path = os.path.join(FIGURES_DIR, "Figure_4.2_Severity_Distribution.pdf")

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')

print(f"\n✅ Figure saved to: {png_path}")
print(f"✅ Figure saved to: {pdf_path}")

# Also display the figure
print("\n📊 Displaying figure...")
plt.show()

print("\n✅ Done!")