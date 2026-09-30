# CLAUDE.md — fork LaurentFrx/Open_Duck_Playground

Entraînement RL (PPO Brax sur MuJoCo MJX / MuJoCo Playground) de la politique de marche d'Open Duck Mini v2.
Fork de `apirrone/Open_Duck_Playground` (remote `upstream`). Dépôt matériel associé : `../Open_Duck_Mini` (branche `v2`).

## Workflow : VPS = code, PC = entraînement

| Machine | Rôle | Interdit |
|---|---|---|
| VPS `feroux-vps` (pas de GPU) | lire, modifier, commiter, pousser sur le fork | installer JAX/CUDA/TensorFlow, lancer `uv run` / `uv sync` (installerait `jax[cuda12]` + `tensorflow`, plusieurs Go), entraîner |
| PC WSL2 (RTX 4070 12 Go VRAM, 32 Go RAM) | `git pull`, installer les dépendances, entraîner, inférer, exporter ONNX | pousser des checkpoints ou des `.onnx` (ignorés par `.gitignore`) |

- Travail sur des branches `laurent/*` ; `main` suit `upstream/main` et ne reçoit aucun commit direct.
- Synchroniser avec l'amont : `git fetch upstream && git switch main && git merge --ff-only upstream/main && git push origin main`.
- Vérification possible sur le VPS sans dépendances : `python3 -m py_compile <fichier>.py`.

## Commande d'entraînement de référence (sur le PC)

À lancer **depuis la racine du dépôt** (chemins relatifs : `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl`, `.tmp/jax_cache`, `checkpoints/`) :

```bash
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
```

Options de `playground/open_duck_mini_v2/runner.py` : `--env joystick|standing` (défaut `joystick`), `--task` (défaut `flat_terrain`),
`--num_timesteps` (défaut 150 M), `--output_dir` (défaut `checkpoints`), `--restore_checkpoint_path`.
Suivi : `uv run tensorboard --logdir=checkpoints`.
À chaque évaluation (`num_evals` = 15), un checkpoint orbax `checkpoints/<date>_<step>` et un `.onnx` sont écrits, plus une copie `ONNX.onnx` dans le répertoire courant (`playground/common/export_onnx.py:177`).

Inférence visuelle d'une politique : `uv run playground/open_duck_mini_v2/mujoco_infer.py -o <fichier.onnx>` (viewer MuJoCo, clavier pour les commandes).

## Contrainte matérielle : 12 Go de VRAM

La config PPO **n'est pas dans ce dépôt** : `playground/common/runner.py:87` reprend
`locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")` de mujoco_playground
(`num_envs=8192`, `batch_size=256`, `num_minibatches=32`, `unroll_length=20`, `num_evals=15`, réseaux 512-256-128).
Aucune option CLI pour `num_envs`.

En cas d'OOM, réduire `num_envs` en ajoutant dans `BaseRunner.train()`, après la ligne `num_timesteps` (`runner.py:101`) :

```python
self.ppo_training_params["num_envs"] = 4096  # puis 2048 si OOM persiste
```

Brax exige `batch_size * num_minibatches % num_envs == 0` (256 × 32 = 8192) : garder `num_envs` parmi 8192, 4096, 2048, 1024.
Autres leviers JAX : `XLA_PYTHON_CLIENT_PREALLOCATE=false` ou `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (JAX préalloue 75 % de la VRAM par défaut).
WSL2 n'accorde par défaut que la moitié de la RAM de l'hôte ; la compilation XLA en consomme beaucoup (réglage `memory=` dans `%UserProfile%\.wslconfig`).

## Structure

```
pyproject.toml                  dépendances (uv) ; uv.lock est ignoré par git
playground/common/
  runner.py                     BaseRunner : config PPO, cache JAX, checkpoints, export ONNX, TensorBoard
  randomize.py                  domain randomization (appliquée à chaque env par Brax)
  rewards.py                    fonctions de récompense génériques (JAX) ; rewards_numpy.py = version NumPy
  poly_reference_motion.py      mouvement de référence polynomial (récompense d'imitation)
  export_onnx.py                conversion des poids Brax → Keras → ONNX (tf2onnx, opset 11)
  onnx_infer.py                 inférence ONNX (onnxruntime)
playground/open_duck_mini_v2/
  runner.py                     point d'entrée d'entraînement (CLI ci-dessus)
  joystick.py                   env principal : suivi d'une commande vx, vy, yaw + tête
  standing.py                   env « debout » (sans imitation)
  base.py                       classe de base MJX : chargement du MJCF, accès joints / backlash / capteurs
  constants.py                  tâche → fichier XML, noms des sites, géométries et capteurs
  custom_rewards.py             récompense d'imitation
  mujoco_infer.py               simulation MuJoCo native d'une politique ONNX (viewer)
  data/polynomial_coefficients.pkl  mouvement de référence (généré par Open_Duck_reference_motion_generator)
  xmls/                         MJCF (voir ci-dessous) + assets/ (STL, hfield.png)
```

## Où régler quoi

| Paramètre | Emplacement |
|---|---|
| Poids des récompenses (`joystick`) | `playground/open_duck_mini_v2/joystick.py:77` (`reward_config.scales`), calcul dans `_get_reward` |
| Poids des récompenses (`standing`) | `playground/open_duck_mini_v2/standing.py:73` |
| Imitation on/off | `USE_IMITATION_REWARD` en tête de `joystick.py` (True) et `standing.py` (False) |
| Domain randomization (frottements, armature, masses, CoM, qpos0, Kp) | `playground/common/randomize.py:26` |
| Bruit capteurs, délais action/IMU, poussées | `joystick.py` : `noise_config` (l. 60), `push_config` (l. 89) |
| Plages de commandes | `joystick.py` : `lin_vel_x`, `lin_vel_y`, `ang_vel_yaw`, plages tête |
| `ctrl_dt` = 0.02 s, `sim_dt` = 0.002 s | `joystick.py:51`, `standing.py:47` ; **dupliqués en dur** dans `mujoco_infer_base.py:15-16` (`sim_dt`, `decimation = 10`) et `mujoco_infer.py:33` (filtre à 50 Hz) |
| `num_envs`, `batch_size`, réseau, `num_evals` | mujoco_playground `config/locomotion_params.py` (voir « Contrainte matérielle ») |
| `num_timesteps` | CLI `--num_timesteps`, appliqué dans `common/runner.py:101` |

## MJCF et tâches

| `--task` | Scène | Robot inclus |
|---|---|---|
| `flat_terrain` | `xmls/scene_flat_terrain.xml` | `open_duck_mini_v2.xml` |
| `flat_terrain_backlash` | `xmls/scene_flat_terrain_backlash.xml` | `open_duck_mini_v2_backlash.xml` (joints passifs modélisant le jeu des servos) |
| `rough_terrain_backlash` | `xmls/scene_rough_terrain_backlash.xml` | `open_duck_mini_v2_backlash.xml` + `assets/hfield.png` |
| `rough_terrain` | `xmls/scene_rough_terrain.xml` — **fichier absent : tâche cassée** | — |

Fichiers inclus : `joints_properties.xml`, `sensors.xml`. `xmls/config.json` = configuration onshape-to-robot ayant servi à générer le MJCF.

## Pièges connus (constatés le 30/09/2026, non corrigés)

- **Dépendance non épinglée** : `playground>=0.0.3` sans lockfile versionné. PyPI propose aujourd'hui `playground` 0.2.0, où
  `mujoco_playground/_src/collision.py` n'existe plus (supprimé depuis 0.1.0) ; `joystick.py` l'importe
  (`from mujoco_playground._src.collision import geoms_colliding`). Le dernier commit amont date du 11/08/2025, époque de 0.0.5.
  Piste avant le premier entraînement : épingler `playground==0.0.5` et vérifier la résolution de `brax` / `mujoco`.
- `mujoco`, `brax`, `flax`, `orbax`, `tensorboardX` ne sont pas déclarés : ils arrivent en transitif via `playground`.
- `README.md` amont cite des fichiers XML qui n'existent plus (`scene_mjx_*`, `open_duck_mini_v2_no_head.xml`).
- Fichier vide `aze` à la racine (reliquat amont).
- Pas de fichier `LICENSE` dans ce dépôt ; 5 sources sur 22 portent un en-tête Apache 2.0 (DeepMind, Antoine Pirrone, Steve Nguyen) :
  `base.py`, `joystick.py`, `standing.py`, `constants.py`, `common/randomize.py`. Le reste n'a aucune mention de licence.
