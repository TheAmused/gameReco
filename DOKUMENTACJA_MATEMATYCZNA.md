# Dokumentacja matematyczna systemu rekomendacji Steam ML

## 1. Cel systemu

System rekomendacji przewiduje, ktore gry Steam moga pasowac uzytkownikowi na podstawie:

1. cech tekstowych gier (tagi i gatunki),
2. historii grania uzytkownika (czas gry),
3. podobienstwa wektorowego miedzy profilem uzytkownika i gra.

Jest to podejscie typu content-based filtering (filtrowanie oparte o zawartosc).

## 2. Architektura i przeplyw danych

Logika biznesowa jest zaimplementowana w [app.py](app.py).

1. Podczas startu aplikacji:
   1. ladowany jest klucz API Steam z pliku .env,
   2. ladowana jest baza [games.json](games.json),
   3. z kazdej gry budowany jest tekst cech,
   4. trenowany jest model TF-IDF.
2. Uzytkownik wpisuje steamID64 na [templates/index.html](templates/index.html).
3. Endpoint /recommend pobiera gry uzytkownika przez Steam API.
4. Z top 10 najdluzej ogrywanych gier tworzony jest profil uzytkownika.
5. Profil porownywany jest do wszystkich gier w bazie przez prawdopodobieństwo cosinusowe.
6. Odfiltrowane sa gry juz posiadane przez uzytkownika.
7. Zwracane jest top 6 rekomendacji do [templates/results.html](templates/results.html).

## 3. Dane wejsciowe i reprezentacja

Kazda gra i z bazy ma miedzy innymi:

1. appid,
2. nazwe,
3. tags,
4. genres,
5. metadata prezentacyjne (obraz, opis, cena).

Wektor cech tekstowych gry powstaje jako konkatenacja:

$$
f_i = \text{join}(\text{tags}_i) + \text{ " " } + \text{join}(\text{genres}_i)
$$

gdzie:

1. $f_i$ to surowy tekst cech gry i,
2. tags_i i genres_i to listy etykiet semantycznych gry.

## 4. Model TF-IDF

Zbior dokumentow to wszystkie teksty cech gier:

$$
\mathcal{D} = \{f_1, f_2, \ldots, f_N\}
$$

Niech:

1. $t$ bedzie tokenem (slowem),
2. $d$ bedzie dokumentem (jedna gra),
3. $N$ liczba wszystkich gier,
4. $\mathrm{df}(t)$ liczba dokumentow zawierajacych token $t$.

Term Frequency:

$$
\mathrm{tf}(t,d) = \text{liczba wystapien } t \text{ w } d
$$

Inverse Document Frequency (intuicyjnie):

$$
\mathrm{idf}(t) = \log\left(\frac{N}{\mathrm{df}(t)}\right)
$$

W praktyce implementacja scikit-learn stosuje wariant wygładzony i normalizacje.

Waga TF-IDF tokenu w dokumencie:

$$
w_{t,d} = \mathrm{tf}(t,d) \cdot \mathrm{idf}(t)
$$

Po transformacji kazda gra i ma wektor:

$$
\mathbf{x}_i \in \mathbb{R}^V
$$

gdzie $V$ to rozmiar slownika tokenow.

W kodzie odpowiada temu:

1. konstruktor: [app.py](app.py#L67),
2. trening i transformacja: [app.py](app.py#L69).

## 5. Budowa profilu uzytkownika

Po pobraniu gier uzytkownika z API:

1. gry sa sortowane po playtime_forever malejaco,
2. wybierane jest top 10,
3. mapowane sa appid na indeksy w macierzy TF-IDF,
4. liczona jest srednia wektorow wybranych gier.

Formalnie, jesli zbior indeksow ulubionych gier to $I_u$:

$$
\mathbf{p}_u = \frac{1}{|I_u|}\sum_{i \in I_u} \mathbf{x}_i
$$

gdzie $\mathbf{p}_u$ to profil uzytkownika.

W kodzie:

1. wybor top gier: [app.py](app.py#L96), [app.py](app.py#L98),
2. mapowanie do indeksow: [app.py](app.py#L101),
3. srednia wektorow: [app.py](app.py#L106).

## 6. Miara podobienstwa: cosine similarity

Dla profilu uzytkownika $\mathbf{p}_u$ i kazdej gry $\mathbf{x}_j$ liczone jest:

$$
\mathrm{sim}(u,j) = \frac{\mathbf{p}_u \cdot \mathbf{x}_j}{\|\mathbf{p}_u\|_2\|\mathbf{x}_j\|_2}
$$

Zakres wartosci:

1. od -1 do 1 dla przestrzeni ogolnej,
2. w tej aplikacji zazwyczaj od 0 do 1 (nieujemne cechy TF-IDF).

Interpretacja:

1. im wyzsza wartosc, tym bardziej podobna gra,
2. wartosc *100 pokazuje procentowe dopasowanie na UI.

W kodzie:

1. obliczenie podobienstw: [app.py](app.py#L108),
2. sortowanie malejace: [app.py](app.py#L110),
3. procent dopasowania: [app.py](app.py#L122).

## 7. Filtrowanie wynikow

System usuwa gry, ktore uzytkownik juz posiada:

$$
R_u = \{j: j \notin O_u\}
$$

gdzie:

1. $O_u$ to zbior appid posiadanych gier,
2. $R_u$ to kandydaci do rekomendacji.

Nastepnie wybierane jest pierwsze $K=6$ gier o najwyzszej podobnosci.

W kodzie:

1. budowa zbioru posiadanych: [app.py](app.py#L100),
2. filtr i agregacja wynikow: [app.py](app.py#L113), [app.py](app.py#L127).

## 8. Opis funkcji i endpointow

### 8.1 Funkcja index

Lokalizacja: [app.py](app.py#L74)

1. Metoda HTTP: GET.
2. Zadanie: zwraca formularz wejscia steamID64.
3. Widok: [templates/index.html](templates/index.html).

### 8.2 Funkcja recommend

Lokalizacja: [app.py](app.py#L78)

1. Metoda HTTP: POST.
2. Wejscie: pole formularza steam_id.
3. Operacje:
   1. walidacja steam_id,
   2. zapytanie do Steam API,
   3. ekstrakcja i sortowanie gier po czasie gry,
   4. konstrukcja profilu uzytkownika,
   5. ranking przez cosine similarity,
   6. filtrowanie gier juz posiadanych,
   7. przekazanie danych do szablonu wynikow.
4. Wyjscie: [templates/results.html](templates/results.html) z listami:
   1. recommendations,
   2. favorite_games.

## 9. Zlozonosc obliczeniowa

Niech:

1. $N$ liczba gier,
2. $V$ rozmiar slownika TF-IDF,
3. $M$ liczba gier uzytkownika,
4. $K$ liczba rekomendacji (tu 6).

Glowne etapy:

1. Budowa TF-IDF (offline przy starcie): w przyblizeniu proporcjonalna do calkowitej liczby tokenow w bazie.
2. Budowa profilu uzytkownika: $O(10 \cdot V)$, bo srednia liczona jest z max 10 wektorow.
3. Podobienstwo do calej bazy: $O(N \cdot V)$.
4. Sortowanie wszystkich wynikow: $O(N \log N)$.

W praktyce etap dominujacy online to porownanie profilu do wszystkich gier i sortowanie.

## 10. Stabilnosc i warunki brzegowe

System obsluguje nastepujace przypadki:

1. brak klucza API -> wyjatek przy starcie,
2. pusta baza games.json -> komunikat bledu,
3. brak publicznego profilu Steam -> brak danych i komunikat,
4. brak mapowania ulubionych gier do bazy lokalnej -> komunikat o braku dopasowania.

Miejsca w kodzie:

1. walidacja klucza: [app.py](app.py#L15),
2. pusta baza: [app.py](app.py#L62), [app.py](app.py#L83),
3. API status: [app.py](app.py#L87),
4. brak gier uzytkownika: [app.py](app.py#L92),
5. brak indeksow top gier: [app.py](app.py#L103).

## 11. Ograniczenia modelu

1. Brak collaborative filtering: system nie korzysta z preferencji innych graczy.
2. Brak wag po playtime: top 10 jest wybierane po czasie, ale srednia wektorow nie jest wazona czasem.
3. Brak ewaluacji metryk rankingowych: nie sa raportowane np. Precision@K, Recall@K, NDCG.
4. Cold-start dla nowych gier i dla niepelnych metadanych.
5. Jakosc rekomendacji zalezy od jakosci i spojnosci tagow w bazie.

## 12. Mozliwe rozszerzenia matematyczne

1. Wazony profil uzytkownika:

$$
\mathbf{p}_u = \frac{\sum_{i \in I_u} \alpha_i \mathbf{x}_i}{\sum_{i \in I_u} \alpha_i}
$$

gdzie $\alpha_i$ moze byc funkcja czasu gry, np. $\alpha_i = \log(1 + \text{hours}_i)$.

2. Zmiana rankingu przez top-n zamiast sortowania pelnego (oszczednosc czasu przy bardzo duzym N).
3. Hybryda content + collaborative filtering.
4. Ewaluacja offline na zbiorze walidacyjnym i testowym.

## 13. Pseudokod algorytmu rekomendacji

```text
INPUT: steam_id, baza gier G z wektorami TF-IDF X

1. Pobierz gry uzytkownika U ze Steam API
2. Posortuj U malejaco po playtime
3. Wez top10 z U jako F
4. Znajdz indeksy I gier z F, ktore istnieja w bazie G
5. Jesli I puste -> zakoncz komunikatem
6. Profil p = mean(X[I])
7. Dla kazdej gry j w G policz sim_j = cosine(p, X[j])
8. Usun gry nalezace do zbioru posiadanych przez uzytkownika
9. Posortuj kandydatow po sim_j malejaco
10. Zwroc pierwsze 6 rekordow

OUTPUT: lista rekomendacji R
```

## 14. Podsumowanie techniczne

System realizuje klasyczny pipeline ML dla rekomendacji tresci:

1. ekstrakcja cech tekstowych,
2. wektoryzacja TF-IDF,
3. budowa profilu przez agregacje wektorow,
4. ranking przez cosine similarity,
5. filtrowanie biznesowe i prezentacja wynikow.

To rozwiazanie jest proste, interpretowalne i dobre jako projekt dydaktyczny na przedmiot Uczenie Maszynowe.