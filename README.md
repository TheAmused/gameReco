
# 🎮 Steam ML Recommender

Aplikacja webowa napisana w języku **Python (Flask)**, wykorzystująca algorytmy **uczenia maszynowego (Machine Learning)** do rekomendowania gier na platformie Steam.

Model analizuje tagi oraz gatunki gier, w które użytkownik grał najdłużej, buduje wektor profilu gracza za pomocą algorytmu **TF-IDF**, a następnie przy użyciu **podobieństwa kosinusowego (Cosine Similarity)** dopasowuje najlepiej pasujące, nieposiadane jeszcze gry z obszernej bazy danych.

---

## 🚀 Instrukcja uruchomienia (krok po kroku)

### 1. Pobranie bazy danych (`games.json`)

Aplikacja do działania na własnym serwerze potrzebuje bazy gier z przypisanymi tagami i statystykami.

1. Wejdź na stronę Hugging Face: [FronkonGames/steam-games-dataset](https://huggingface.co/datasets/FronkonGames/steam-games-dataset/blob/main/games.json)
2. Pobierz plik `games.json` (skorzystaj z przycisku pobierania obok rozmiaru pliku).
3. Umieść pobrany plik **bezpośrednio w głównym folderze projektu** (tam, gdzie znajduje się plik `app.py`).

### 2. Utworzenie środowiska wirtualnego (venv)

**Windows:**

```bash
py -m venv venv
venv\Scripts\activate
```

*(Po poprawnej aktywacji, na początku linii w terminalu powinien pojawić się napis `(venv)`).*

### 3. Instalacja bibliotek (`requirements.txt`)

```bash
py -m pip install -r requirements.txt
```

### 4. Wygenerowanie klucza Steam API i konfiguracja (`.env`)

Aby pobierać dane o koncie gracza, potrzebujesz darmowego klucza API od Valve.

1. Zaloguj się na swoje konto Steam i przejdź tutaj: [Steam API Keys](https://steamcommunity.com/dev?l=polish)
2. Wygeneruj klucz.
3. W głównym folderze projektu utwórz plik o nazwie `.env` (z kropką na początku!).
4. Wklej do niego wygenerowany klucz w formacie:

```env
STEAM_API_KEY=TUTAJ_WKLEJ_SWOJ_WYGENEROWANY_KLUCZ
```

### 5. Uruchomienie aplikacji

Gdy środowisko jest aktywne, a wszystkie pliki są na swoim miejscu, wystartuj serwer poleceniem:

```bash
py app.py
```

Aplikacja powinna być dostępna w przeglądarce pod adresem: **http://127.0.0.1:5000**

---

## 🕹️ Jak korzystać z aplikacji?

Aby sprawdzić rekomendacje dla danego gracza, potrzebujesz jego unikalnego identyfikatora **steamID64**. Zwykły nick ze Steama nie zadziała!

1. Wejdź na stronę:[https://steamid.io/](https://steamid.io/)
2. Wklej tam link do profilu Steam, dla którego chcesz wygenerować rekomendacje.
3. Skopiuj wartość oznaczoną jako **`steamID64`** (jest to ciąg 17 cyfr).
4. Wklej skopiowany numer w formularzu na stronie głównej uruchomionej aplikacji.

> **⚠️ UWAGA:** Aby aplikacja zadziałała prawidłowo, konto Steam musi mieć **publiczne ustawienia prywatności** (w tym w szczególności sekcję "Szczegóły gier / Game details"). W przeciwnym razie Steam API nie udostępni informacji o czasie gry i aplikacja zwróci błąd.
