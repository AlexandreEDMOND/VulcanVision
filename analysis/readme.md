# Roadmap & notes Vesuvius Challenge

## État des lieux rapide
- Données : `train_images`/`train_labels` = 806 volumes, `test_images` = 1 volume ; shapes dominantes 320³ (757), 256³ (48), 384³ (1).
- Cohérence CSV/fichiers : OK (correspondance parfaite).
- Voir `notebooks/baseline_unet3d.ipynb` : UNet 3D simple, split interne train/val/test, métriques SurfaceDice/VOI approximées, courbes de loss, inférence sur test officiel.

## Plan projet (étapes)
- **Phase 0 – Setup** : verrouiller l’environnement (torch+tifffile+scipy), vérifier GPU, préparer config (paths, patch_size, epochs) et logging simple (CSV).
- **Phase 1 – Baseline** : entraîner le UNet3D léger sur un subset, valider avec SurfaceDice/VOI internes, générer `submission.zip` pour un premier score public.
- **Phase 2 – Validation robuste** : split par `scroll_id` (éviter fuite), holdout fixe, logger métriques par run, checkpoints modèles.
- **Phase 3 – Préprocessing** : normalisation par volume, padding propre des tailles variables, tuilage avec overlap + fenêtrage pour lisser les bords, sampler focalisé sur zones positives.
- **Phase 4 – Modèle & pertes** : UNet3D plus large (attention/SE), pertes combinées BCE+Dice+clDice/skeleton, aug 3D légères (flip, bruit, elastic soft).
- **Phase 5 – Inférence & postprocess** : seuil optimisé sur val, morpho légère (open/close), blending overlap, gérer mémoire GPU pour gros volumes.
- **Phase 6 – Itération Kaggle** : soumissions régulières, comparer scores internes vs LB, ajuster stride/patch_size/epochs et architecture selon retours.

## Pistes d’amélioration
- **Topologie** : intégrer TopoScore (lib topo type gudhi/persim si dispo) ou proxy (clDice, skeleton loss) pour réduire ponts/ruptures.
- **Sampling** : hard-example mining sur bords, mix de patches centrés sur positifs + aléatoires pour équilibre.
- **Tuilage** : overlap + fenêtre (Hann) pour réduire artefacts de couture ; stride plus petit en inférence si budget temps.
- **Augmentations** : flips, légers décalages d’intensité, bruit gaussien/poisson, petites elastic transform 3D contrôlées.
- **Normalisation** : min-max par volume, éventuellement z-score local ; tester histogram clipping pour limiter les outliers.
- **Architecture** : UNet3D avec canaux plus larges, residual/attention blocks, small 3D Swin/ConvNeXt si budget GPU.
- **Postprocess** : suppression de petites composantes, fermeture des trous fins, seuil adaptatif via val.

## Out of the Box
- **Multi-vues 2.5D** : générer des coupes multi-orientations (axial/coronal/sagittal) et agréger les prédictions 2.5D pour un compromis mémoire/précision.
- **Graph/topology aware** : convertir la surface prédite en graph/squelette et appliquer du message passing (GNN) pour corriger ponts/ruptures avant back-projection.
- **Noise2Noise/denoising prétexte** : pré-entraîner un débruiteur 3D (DnCNN/UNet) sur volumes bruyants puis fine-tune en segmentation (meilleure robustesse).
- **Distillation volumique** : entraîner un modèle plus grand offline, distiller vers un petit UNet3D pour soumission rapide.
- **Curriculum spatial** : commencer sur patches centrés sur zones faciles (fibres claires), élargir progressivement vers zones compressées/tordues.
