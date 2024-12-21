import spacy
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from flask import Flask, request, render_template_string

# Télécharger les ressources nécessaires pour NLTK (VADER lexicon)
nltk.download('vader_lexicon')

# Initialisation du modèle spaCy pour l'anglais, c'est celui que j'utilise pour analyser la structure grammaticale du texte
nlp = spacy.load('en_core_web_sm')

# Initialisation de VADER pour l'analyse du sentiment
sia = SentimentIntensityAnalyzer()

# Fonction de nettoyage du texte
def clean_text(text):
    # 1. Enlever les caractères spéciaux et convertir tout en minuscules pour normaliser
    text = re.sub(r'[^\w\s]', '', text.lower())

    # 2. Analyser le texte avec spaCy pour obtenir des informations grammaticales
    doc = nlp(text)
    
    # 3. Liste des mots qui peuvent inverser le sentiment (ex : 'not', 'never', 'no', etc.)
    sentiment_inverter = ['not', 'never', 'no', 'without']

    for i, token in enumerate(doc):
        # 4. Si on trouve un mot qui inverse le sentiment comme "not", "never", etc.
        if token.text in sentiment_inverter and i + 1 < len(doc):
            # 5. On vérifie si le mot suivant est un adjectif ou un verbe qui pourrait être affecté par l'inversion
            next_token = doc[i + 1]
            
            if next_token.dep_ in ['amod', 'acomp', 'xcomp']:  # Adjectifs ou compléments verbaux
                # 6. On marque l'inversion du sentiment en ajoutant 'neg_' avant le mot suivant
                text = text[:text.find(next_token.text)] + 'neg_' + next_token.text + text[text.find(next_token.text) + len(next_token.text):]
    
    return text

# Fonction qui analyse le sentiment d'un texte
def analyze_sentiment(text):
    # 1. Nettoyer le texte d'abord pour enlever les caractères spéciaux et le normaliser
    cleaned_text = clean_text(text)

    # 2. Utilisation de VADER pour analyser le sentiment du texte
    sentiment_scores = sia.polarity_scores(cleaned_text)
    
    # 3. Extraction du score global du sentiment ('compound' score)
    sentiment_value = sentiment_scores['compound']

    # 4. Déterminer le sentiment final en fonction du score
    if sentiment_value >= 0.05:
        sentiment = 'positive'
    elif sentiment_value <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return sentiment


# Initialisation de l'application Flask
app = Flask(__name__)

# Route principale de l'application
@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    if request.method == 'POST':  # Si un avis est soumis via POST
        review = request.form['review']  # Récupérer l'avis soumis par l'utilisateur

        # Analyser le sentiment de l'avis
        sentiment = analyze_sentiment(review)

    # Code HTML pour l'interface utilisateur
    return render_template_string(f'''
    <!DOCTYPE html>
    <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Sentiment Analyzer</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f5f5f5;
                    color: #febd69;
                    text-align: center;
                    padding: 20px;
                }}
                h1 {{
                    color: #132f8c;
                }}
                textarea {{
                    width: 80%;
                    height: 100px;
                    margin: 10px 0;
                    padding: 10px;
                    font-size: 16px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                input[type="submit"] {{
                    padding: 10px 20px;
                    font-size: 16px;
                    background-color: #febd69;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                }}
                input[type="submit"]:hover {{
                    background-color: #febd69;
                }}
                h2 {{
                    margin-top: 20px;
                    color: #132f8c;
                }}
                h3 {{
                    color: #febd69;
                }}
            </style>
        </head>
        <body>

            <h1>Sentiment Analyzer for Reviews</h1>
            <form method="POST">
                <textarea name="review" placeholder="What do you think about the product?"></textarea><br>
                <input type="submit" value="Analyze">
            </form>
            <h2>Your review is: <h3>{sentiment if sentiment else ""}</h3></h2>
        </body>
    </html>
    ''')

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
