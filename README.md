# Projet de TAILOR AI

## Présentation du Projet
Ce projet, conçu par **NANI CHARLY** & **DOHA CADNEL**, propose une solution innovante alliant la tradition de la couture sur mesure aux technologies de pointe. L'objectif est de permettre aux utilisateurs de prendre leurs mesures de manière autonome, précise et à distance.

## Problématique
Comment offrir une expérience de prise de mesures fiable et sans contact afin de commander des vêtements personnalisés en ligne, tout en réduisant les erreurs manuelles et les retours de marchandises liés aux problèmes de taille ?

## Objectifs Principaux
*   **Acquisition d'images :** Permettre le chargement d'images 2D (photos) via une interface dédiée.
*   **Détection de points clés :** Identifier automatiquement les articulations et points morphologiques essentiels (épaules, taille, hanches, etc.) grâce à la vision par ordinateur.
*   **Calcul des mesures :** Générer les dimensions corporelles précises adaptées à la confection de vêtements, indépendamment de la morphologie de l'utilisateur.

## Choix Techniques
*   **Vision par Ordinateur :** Utilisation de l'architecture **CNN (Réseaux de Neurones Convolutifs)** basée sur la famille de modèles **OpenPose**.
*   **Traitement d'Image :** Exploitation des bibliothèques **OpenCV** et **PIL** pour le prétraitement et la normalisation des données.
*   **Modèle d'IA :** Utilisation du framework **Caffe** et de la base de données **COCO** (17 points clés étiquetés) pour une précision optimale.
*   **Algorithmes :** Utilisation de cartes de chaleur (*Heatmaps*) pour localiser les points clés avec une probabilité maximale.

## Avantages de la Solution
*   **Précision accrue :** Élimination des erreurs humaines lors de la prise de mesure.
*   **Expérience client :** Gain de temps, confort et accès à des tailleurs internationaux sans déplacement.
*   **Optimisation industrielle :** Réduction significative du taux de retour des vêtements pour les e-commerçants.
