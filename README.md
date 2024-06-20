### Instalacja zależności
Wszelkie zależności znajdują się w pliku `pyproject.toml`

Projekt został stworzony za pomocą wirtualnego środowiska `Poetry` i w ten sposób najlepiej go odpalać:
```bash
poetry shell
poetry install
poetry run python -m ...
```

By odpalić poszczególne moduły:

```bash
poetry run python -m {moduł}
```

### Moduły:
Wszelkie dane stworzone przez poniższe moduły będą pojawiać się w folderze `./data`

 Pobieranie danych:

W pliku `.env` w folderze `scrap` winny znajdować się loginy i hasła do kont email/twitter zgodnie ze strukturą w pliku `.env.example`

`src.scrap.scrap_twitter`

 Preprocessing danych:

 `src.preprocess.preprocess`

Tworzenie Bag of words:

`src.words.bow`

Tworzenie Word Clouds:

`src.words.cow`

Topic modelling:

`src.topic_modelling.tm`

Analiza sentymentu:

`src.sentiment.nltk_sentiment`