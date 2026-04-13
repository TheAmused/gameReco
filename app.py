import os
import json
from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

STEAM_API_KEY = os.getenv('STEAM_API_KEY')

if not STEAM_API_KEY:
    raise ValueError("Brak klucza STEAM_API_KEY! Upewnij się, że plik .env istnieje i zawiera klucz.")

print("Ładowanie bazy gier games.json...")

games_data =[]
dataset = {}

if os.path.exists('games.json'):
    with open('games.json', 'r', encoding='utf-8') as fin:
        text = fin.read()
        if len(text) > 0:
            dataset = json.loads(text)

for appID_str, game in dataset.items():
    try:
        appID = int(appID_str)
        name = game.get('name', 'Brak nazwy')
        tags_raw = game.get('tags',[])
        
        if isinstance(tags_raw, dict):
            tags_list = list(tags_raw.keys())
        elif isinstance(tags_raw, list):
            tags_list = tags_raw
        else:
            tags_list = []
            
        tags_str = " ".join([str(t) for t in tags_list])
        
        genres_raw = game.get('genres',[])
        genres_str = " ".join([str(g) for g in genres_raw])
        
        combined_features = tags_str + " " + genres_str

        if combined_features.strip():
            games_data.append({
                'appid': appID,
                'name': name,
                'features': combined_features,
                'header_image': game.get('header_image', ''),
                'short_desc': game.get('short_description', ''),
                'price': game.get('price', 0.0),
                'display_tags': ", ".join(tags_list[:5])
            })
    except Exception as e:
        continue

if not games_data:
    print("UWAGA: Nie wczytano żadnych gier! Upewnij się, że plik games.json jest poprawny.")

df_games = pd.DataFrame(games_data)

print(f"Załadowano {len(df_games)} gier. Trenowanie modelu TF-IDF...")

tfidf = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words='english')
if not df_games.empty:
    tfidf_matrix = tfidf.fit_transform(df_games['features'])
    print("Model ML gotowy!")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    steam_id = request.form.get('steam_id')
    
    if not steam_id:
        return "Błąd: Nie podano Steam ID."
        
    if df_games.empty:
        return "Błąd serwera: Baza gier (games.json) jest pusta lub nie została wczytana."

    url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={STEAM_API_KEY}&steamid={steam_id}&include_appinfo=1&format=json"
    response = requests.get(url)
    
    if response.status_code != 200:
        return "Błąd komunikacji z API Steam. Sprawdź swój klucz API w pliku .env i upewnij się, że profil gracza jest publiczny."
        
    data = response.json()
    games = data.get('response', {}).get('games',[])
    
    if not games:
        return "Nie znaleziono gier. Upewnij się, że profil (i 'Szczegóły gry') na Steam są ustawione jako publiczne."

    top_played = sorted(games, key=lambda x: x['playtime_forever'], reverse=True)
    
    top_10_games = top_played[:10]
    top_appids = [g['appid'] for g in top_10_games]
    owned_appids = [g['appid'] for g in games]

    user_indices = df_games[df_games['appid'].isin(top_appids)].index
    
    if len(user_indices) == 0:
        return "Niestety, żadnej z Twoich ulubionych gier nie ma w naszej bazie games.json."

    user_profile = np.asarray(tfidf_matrix[user_indices].mean(axis=0))

    similarities = cosine_similarity(user_profile, tfidf_matrix)
    
    similar_indices = similarities.argsort()[0][::-1]

    recommendations =[]
    for idx in similar_indices:
        game_appid = df_games.iloc[idx]['appid']
        
        if game_appid not in owned_appids:
            recommendations.append({
                'name': df_games.iloc[idx]['name'],
                'header_image': df_games.iloc[idx]['header_image'],
                'short_desc': df_games.iloc[idx]['short_desc'],
                'price': df_games.iloc[idx]['price'],
                'tags': df_games.iloc[idx]['display_tags'],
                'similarity': round(similarities[0][idx] * 100, 1)
            })
        
        if len(recommendations) >= 6: 
            break

    favorite_games_info =[{'name': g['name'], 'playtime_hours': round(g['playtime_forever']/60, 1)} for g in top_10_games[:5]]

    return render_template('results.html', recommendations=recommendations, favorite_games=favorite_games_info)

if __name__ == '__main__':
    app.run(debug=True)