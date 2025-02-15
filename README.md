# Green-AIHack
# Détection d'Images Normales et Anormales gràce à un modèle

## Description du Projet

Ce projet vise à développer un modèle pour détecter des images normales et anormales. Nous avons utilisé un autoencodeur convolutif pour encoder les images et les reconstruire. Le modèle est formé pour minimiser la différence entre les images d'entrée et les images reconstruites. Les images présentant des erreurs de reconstruction élevées sont classées comme anormales.


## Structure du Projet

Le projet est organisé comme suit :


## Données

Les données d'entrées se composent d'images classées en deux catégories : "Normal" et "Anomaly". Chaque catégorie contient des sous-dossiers représentant différentes classes d'images.

## Modèle

Nous avons utilisé un autoencodeur convolutif composé d'un encodeur et d'un décodeur. L'encodeur compresse les images d'entrée en une représentation compacte (bottleneck), et le décodeur reconstruit les images à partir de cette représentation.

## Entraînement

L'entraînement se fait en utilisant une fonction de perte qui mesure l'erreur de reconstruction (erreur quadratique moyenne, MSE). Le modèle est entraîné pour minimiser cette erreur. Le code d'entraînement se trouve dans `train_autoencoder.py`.

## Détection d'Anomalies

Pour détecter les anomalies, nous comparons l'erreur de reconstruction d'une image avec un seuil prédéfini. Si l'erreur dépasse ce seuil, l'image est classée comme anormale. Le code de détection d'anomalies se trouve dans `test_auto.py`.

## Performance

La performance du modèle est évaluée à l'aide des métriques de précision, rappel, score F1, précision globale et de l'empreinte carbonne. Ces métriques permettent de mesurer la capacité du modèle à détecter correctement les anomalies tout en minimisant les faux positifs et les faux négatifs.

## Prérequis

- Python 
- torch
- torchvision
- PIL (Pillow)
- tranformers
- Hugging face
- - CodeCarbon

## Installation

Clonez ce dépôt :
```bash
git clone https://github.com/votre-utilisateur/votre-repo.git
- Branch "moussa" créé pour le test du modèle dans le main actuel mais avec une plus grande partie des éléments de la data set choisie (100%)

## Entraînement

- python train_auto.py


