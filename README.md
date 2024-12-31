# Objectif du Projet

Ce projet vise à réimplémenter l'architecture proposée dans l'article intitulé ["A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multi-Pitch Estimation"](https://arxiv.org/abs/2203.09893).

Ce modèle a été conçu pour la transcription automatique des notes et l'estimation multi-pitch, en utilisant une approche instrument-agnostique, c'est-à-dire qu'il peut fonctionner avec différents types d'instruments sans nécessiter de modèles spécifiques.

## Résumé de l'Architecture

Le modèle est une architecture légère basée sur des réseaux neuronaux, exploitant les techniques suivantes :

- **Harmonic Stacking** pour représenter les données audio de manière compacte et informative.
- Trois sorties principales générées par le modèle :
  - **Yp** : Une estimation brute multi-pitch pour capturer les fréquences présentes.
  - **Yn** : Une représentation quantifiée pour détecter les événements de note.
  - **Yo** : Une estimation des onsets (début des notes), essentielle pour la transcription.

## Choix de la Base de Données

Dans l'article original, l'architecture a été entraînée sur cinq bases de données et (7 bases de données au total utilisés avec les tests). Cependant, pour ce projet, nous avons décidé de nous concentrer uniquement sur la base de données **GuitarSet**, pour plusieurs raisons :

- **Caractéristiques variées** : GuitarSet contient des enregistrements d'audios mono et polyphoniques, ce qui est idéal pour évaluer un modèle instrument-agnostique.
- **Annotations riches** : GuitarSet fournit des annotations précises à la fois pour les notes et pour les multi-pitch, ce qui permet de générer les matrices cibles nécessaires (Yn, Yo et Yp).
- **Utilisation dans l'article** : Cette base a été utilisée à la fois pour l'entraînement et le test dans l'architecture de l'article, ce qui la rend essentielle pour reproduire leurs résultats.

## Préparation de la Base de Données

Le dossier **DataSet** contient tout le nécessaire pour télécharger, organiser et prétraiter les données GuitarSet. Voici les étapes pour configurer cette base avant l'entraînement :

### Téléchargement des données

- GuitarSet can be downloaded from [Zenodo](https://zenodo.org/record/3371780).
- Téléchargez l'ensemble des fichiers audio "audio_hex-pickup_debleeded" et des annotations correspondantes "annotations".

### Prétraitement des données

Les scripts fournis dans le dossier **DataSet** permettent d'extraire les annotations utiles (notes et multi-pitch). Ces annotations sont utilisées pour générer les matrices binaires cibles suivantes :

- **Yn** : Correspond aux activations de notes (si une note est active).
- **Yo** : Correspond aux onsets des notes.
- **Yp** : Correspond aux activations des pitchs (f0).

### Organisation des données

Les fichiers audio et leurs annotations sont automatiquement organisés en ensembles d'entraînement et de test. Des fichiers `.npy` sont générés pour représenter les matrices cibles Yn, Yo et Yp, prêtes à être utilisées pour l'entraînement du modèle.
