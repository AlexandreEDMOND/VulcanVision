## Résumé du sujet Vesuvius Challenge - Surface Detection

### Objectif
- Segmenter la surface de papyrus dans des scans CT 3D pour permettre le dépliage virtuel de rouleaux carbonisés d'Herculanum.
- La priorité est de suivre la surface (idéalement le recto tourné vers l’ombilic) en évitant fusions entre couches et ruptures topologiques.

### Données
- Chunks 3D de scans CT binarisés, dimensions variables, acquis aux synchrotrons ESRF (Grenoble, BM18) et DLS (Oxford, I12).
- Labels initiaux très soignés, données supplémentaires moins curatorées prévues en cours de compétition.
- Sortie attendue : un masque volume `.tif` par image test, mêmes dimensions et type que la source.

### Évaluation (score dans [0,1], plus haut = mieux)
- Score = 0,30 TopoScore + 0,35 SurfaceDice@τ + 0,35 VOI_score, avec τ = 2.0 (en unités d’espacement).
- SurfaceDice@τ : proximité des surfaces.
- VOI_score : cohérence des composantes (pénalise splits/merges).
- TopoScore : préservation de la topologie (composantes, tunnels, cavités).

### Contraintes de soumission
- Soumettre `submission.zip` contenant un `.tif` nommé `[image_id].tif` par volume test.
- Exécution via notebooks Kaggle ; limites : CPU ou GPU ≤ 9 h, internet désactivé ; données externes publiques autorisées.

### Calendrier
- Début : 13 nov. 2025 ; fin des inscriptions et fusions d’équipes : 6 fév. 2026 ; soumission finale : 13 fév. 2026 à 23:59 UTC. L’organisateur peut ajuster les dates.

### Récompenses
- 1er : 25k$, 2e : 20k$, 3e : 15k$, 4e : 10k$, 5e–10e : 5k$.

### Points d’attention techniques
- Éviter ponts artificiels entre couches et trous inutiles ; favoriser la continuité d’une même couche.
- Penser topologie (composantes, tunnels, cavités) et cohérence d’instance, pas seulement le Dice voxel.
- Le calcul du score complet peut être long ; prévoir du temps pour l’évaluation.

### Ressources utiles
- Pipeline d’unwrapping : https://www.youtube.com/watch?v=yHbpVcGD06U
- Notebook et implémentation du métrique : voir ressources Kaggle de la compétition.

### Exploration rapide des données
- Lancer `python3 scripts/explore_dataset.py` pour générer `analysis/readme.md` avec les observations (comptes d’ids, cohérence fichiers, formes des volumes).

### Notebook de baseline (torch)
- `notebooks/baseline_unet3d.ipynb` : entraînement d’un petit UNet 3D sur un échantillon de patches + split interne train/val/test, métriques proches compétition (SurfaceDice/VOI) sur val et test internes, visualisation des courbes de loss, puis inférence sur le `test_images` officiel pour générer `submission.zip`. Ajuste patch_size, nb d’ids, epochs et stride pour un vrai run GPU. Cellule d’install incluse (`pip install ...` si besoin).
