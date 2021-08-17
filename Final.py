import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

DATA_URL = 'stats.csv'
st.title('NBA 2020-21 Statistics Dashboard')
st.markdown('Our dashboard can be used to compare '
            'statistics from the most recent NBA season')


@st.cache(persist=True)
def load_data():
    salaries = pd.read_csv('salaries.csv')
    stats = pd.read_csv('stats.csv')
    stats['Player'] = [x.split('\\')[0] for x in stats['Player'].tolist()]
    salaries['Player_Name'] = [x.split('\\')[0] for x in salaries['Player_Name'].tolist()]
    salaries = salaries.drop(columns=['Rk', 'Tm', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26', 'Signed Using', 'Guaranteed'])
    combined = pd.merge(stats, salaries, left_on='Player', right_on='Player_Name')
    combined = combined.drop_duplicates(subset=['Player'])
    combined = combined.drop(columns=['Player_Name', 'Rk'])
    combined = combined.set_index('Player')
    combined = combined.rename(columns={'2020-21': '2021 Salary'})
    combined['2021 Salary'] = combined['2021 Salary'].str.strip('$')
    combined['2021 Salary'] = pd.to_numeric(combined['2021 Salary'])
    return combined


stats = load_data()


def correlation_matrix():
    # 3P% to FT%
    # EFG% to FT%
    # TS% to P/36
    # TS% to PER
    # MIN to FTA
    # 3PAr to FG%/EFG%/TS%

    st.header('Correlation Heatmap across various statistics')
    min_games_played = st.slider('Filter # GP:', 1, int(stats['G'].max()))
    filtered_data = stats[stats['G'] >= min_games_played]
    st.markdown(f'Number of players who played at least {min_games_played} games: {len(filtered_data)}')

    data = {'3P%': stats['3P%'].values.tolist(),
            '2P%': stats['2P%'].values.tolist(),
            'FT%': stats['FT%'].values.tolist(),
            # 'FTA': stats['FTA'].values.tolist(),
            'eFG%': stats['eFG%'].values.tolist(),
            'PTS': stats['PTS'].values.tolist(),
            'TRB': stats['TRB'].values.tolist(),
            'AST': stats['AST'].values.tolist(),
            'BLK': stats['BLK'].values.tolist(),
            'TOV': stats['TOV'].values.tolist(),
            'PF': stats['PF'].values.tolist(),
            # 'MP': stats['MP'].values.tolist(),
            # '2021 Salary': stats['2021 Salary'].values.tolist()
            }
    print(data)

    # Creating the correlation matrix
    df = pd.DataFrame(filtered_data, columns=data.keys())
    corrMatrix = df.corr()

    # Creating the correlation heatmap and plotting/showing the results
    fig, ax = plt.subplots()
    mask = np.triu(np.ones_like(df.corr()))
    ax = sns.heatmap(corrMatrix, annot=True, mask=mask, cmap='Oranges')
    st.pyplot(fig)


def bar_chart(player):
    labels = ['Points', 'Rebounds', 'Assists', 'Blocks', 'Turnovers', 'Salary']
    league_average = [stats['PTS'].mean(), stats['TRB'].mean(), stats['AST'].mean(), stats['BLK'].mean(), stats['TOV'].mean(), stats['2021 Salary'].mean()/stats['2021 Salary'].mean()]
    player_stats = [stats.loc[player]['PTS'], stats.loc[player]['TRB'], stats.loc[player]['AST'], stats.loc[player]['BLK'], stats.loc[player]['TOV'], stats.loc[player]['2021 Salary']/stats['2021 Salary'].mean()]

    x = np.arange(len(labels))

    plt.bar(x - 0.35/2, player_stats, label= player + ' Statistcs', width=0.35)
    plt.bar(x + 0.35/2, league_average, label='NBA League Average', width=0.35)
    plt.xticks(x, labels)
    plt.title(player + ' per game statistics against league average')
    plt.legend()
    return plt



def main():
    correlation_matrix()
    chosen_player = st.selectbox("Player", stats.index.tolist())
    st.markdown('All values below are on a per game basis excluding the salary column. The salary statistic is shown on a relative basis where the value 1 indicated the league average of $8.14 million')
    st.pyplot(bar_chart(chosen_player))



main()
