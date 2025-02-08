import json
import numpy as np
import matplotlib.pyplot as plt
import re

def extract_scores(text):
    # Extract Yes/No scores from response text
    match = re.search(r'ANS: \w+, Yes: (\d+)%, No: (\d+)%', text)
    if match:
        yes_score = int(match.group(1))
        no_score = int(match.group(2))
        return yes_score, no_score
    return None

# Initialize lists to store scores for each category
safe_yes = []
safe_no = []
temp_unsafe_yes = []
temp_unsafe_no = []
frame_unsafe_yes = []
frame_unsafe_no = []

# Read and parse the JSON file
with open('responses.json', 'r') as f:
    entries = json.load(f)

# Process each entry
for entry in entries:
    if 'final_response' in entry:
        scores = extract_scores(entry['final_response'])
        if scores:
            yes_score, no_score = scores
            if entry['label'] == 'safe':
                safe_yes.append(yes_score)
                safe_no.append(no_score)
            elif entry['label'] == 'temp_unsafe':
                temp_unsafe_yes.append(yes_score)
                temp_unsafe_no.append(no_score)
            elif entry['label'] == 'frame_unsafe':
                frame_unsafe_yes.append(yes_score)
                frame_unsafe_no.append(no_score)

# Calculate averages
safe_avg = (np.mean(safe_yes), np.mean(safe_no))
temp_unsafe_avg = (np.mean(temp_unsafe_yes), np.mean(temp_unsafe_no))
frame_unsafe_avg = (np.mean(frame_unsafe_yes), np.mean(frame_unsafe_no))

# Print the averages
print("Safe averages (Yes/No):", safe_avg)
print("Temp unsafe averages (Yes/No):", temp_unsafe_avg)
print("Frame unsafe averages (Yes/No):", frame_unsafe_avg)

# Create the visualization
labels = ['Safe', 'Temp Unsafe', 'Frame Unsafe']
yes_means = [safe_avg[0], temp_unsafe_avg[0], frame_unsafe_avg[0]]
no_means = [safe_avg[1], temp_unsafe_avg[1], frame_unsafe_avg[1]]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, yes_means, width, label='Yes Score', color='#8884d8')
rects2 = ax.bar(x + width/2, no_means, width, label='No Score', color='#82ca9d')

# Customize the plot
ax.set_ylabel('Average Score (%)')
ax.set_title('Average Yes/No Scores by Label Category')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add grid lines
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Add value labels on the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(rect.get_x() + rect.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

# Adjust layout and save
plt.tight_layout()
plt.savefig('nsfw_scores.png', dpi=300, bbox_inches='tight')
plt.close()