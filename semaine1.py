import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("male_players.csv")

df = df[[
    'Name', 'Position', 'Age', 'OVR',
    'Sprint Speed', 'Stamina', 'Short Passing',
    'Acceleration', 'Strength', 'Reactions'
]]

df = df.rename(columns={
    'Sprint Speed': 'Speed',
    'Stamina': 'Fatigue',
    'Short Passing': 'PassAccuracy',
})

df_outfield = df[df['Position'] != 'GK'].copy()

plt.figure(figsize=(10, 5))
sns.histplot(df_outfield['Speed'], bins=30, color='blue')
plt.title('Distribution de la vitesse des joueurs')
plt.xlabel('Speed')
plt.savefig('distribution_speed.png')
plt.show()

df_grouped = df_outfield.groupby('Position')[['Fatigue', 'OVR']].mean().reset_index()
df_melted = df_grouped.melt(id_vars='Position', value_vars=['Fatigue', 'OVR'], 
                             var_name='Metric', value_name='Value')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Position', y='Value', hue='Metric', palette='coolwarm')
plt.title('Fatigue vs Performance moyenne par position')
plt.xlabel('Position')
plt.ylabel('Score moyen')
plt.xticks(rotation=45)
plt.legend(title='Métrique')
plt.tight_layout()
plt.savefig('fatigue_vs_ovr.png')
plt.show()

top_speed = df_outfield.nlargest(10, 'Speed')[['Name', 'Speed', 'Position']]
plt.figure(figsize=(10, 5))
sns.barplot(data=top_speed, x='Speed', y='Name', palette='Reds_r')
plt.title('Top 10 joueurs les plus rapides')
plt.savefig('top10_speed.png')
plt.show()

df_outfield.to_csv('dataset_football_propre.csv', index=False)