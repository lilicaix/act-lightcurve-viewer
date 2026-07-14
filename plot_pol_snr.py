import pandas as pd
import matplotlib.pyplot as plt

# load the newly integrated dataset
df = pd.read_csv('all_detected_flares.csv')

# create scatter plot
plt.figure(figsize=(9, 6))

# loop through each unique frequency band to color code
for band in df['frequency'].unique():
    subset = df[df['frequency'] == band]
    plt.scatter(subset['snr'], subset['max_pol_fraction'], alpha=0.7, edgecolors='w', s=60, label=band)

# format plot
plt.title('Max Pol Fraction (3$\sigma$ Upper Limit) vs. Intensity SNR')
plt.xlabel('Intensity SNR')
plt.ylabel('Max Pol Fraction')
plt.legend(title='Band')
plt.grid(True, linestyle='--', alpha=0.6)

# save and show output
plt.savefig('pol_upper_limit_vs_snr.png', dpi=300, bbox_inches='tight')
plt.show()