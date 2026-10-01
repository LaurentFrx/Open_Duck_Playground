# Canard BiscaLab — dossier de travail Open Duck Mini V2

Dossier de travail (PWA) du projet de reproduction de l'Open Duck Mini V2 (Antoine Pirrone, Bordeaux) au BiscaLab. Version 1.0 du 1er octobre 2026.

## Contenu de l'archive
- `docs/index.html` — la PWA complète (page unique, autonome, thème clair de Domo + thème sombre). Le dossier s'appelle `docs/` parce que GitHub Pages sert ce dossier directement depuis la branche `main`.
- `docs/manifest.webmanifest`, `docs/sw.js`, `docs/icons/` — manifeste, service worker (coquille en cache, réseau d'abord), icônes.
- `src/` — les sources de la page (une partie par rubrique), assemblées par `build.py`.
- `recherche/recherche-*.md` — les trois rapports de recherche documentés (dépôts/BOM/runtime ; marche/calibration/sim2real ; organisation collaborative).

## Hébergement
GitHub Pages : Settings → Pages → Deploy from a branch → `main` / `/docs`. L'app est alors à `https://<compte>.github.io/<dépôt>/` et s'installe comme une PWA (chemins relatifs, `scope: ./`). Tout autre hébergeur statique en HTTPS convient aussi. Aucun build, aucune dépendance serveur. Les polices Inter et JetBrains Mono viennent de Google Fonts ; l'app fonctionne sans (repli système).

## Mise à jour
Modifier un fichier de `src/`, lancer `python3 build.py`, pousser (`docs/` est régénéré). Incrémenter `VERSION` dans `sw.js` pour invalider le cache des appareils installés.

## Licence
Contenu rédigé pour le BiscaLab. Le projet Open Duck Mini est sous licence Apache-2.0 (© Antoine Pirrone et contributeurs).
