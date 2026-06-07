# Architecture de roblox-AI

## Vue d'ensemble

Moteur IA multimodal natif pour création de jeux Roblox Studio.

```
Entrée texte → [Modèle Multimodal] → Sorties (3D, Code, Audio, Actions MCP)
                         ↓
              [Agentic Harness] → Roblox Studio (via MCP)
```

## Flux de création d'un jeu

```
1. User décrit le jeu ("Crée un obby médiéval avec 10 obstacles")
2. Text Encoder → espace latent multimodal
3. Fusion Transformer raisonne (spatial + game design + code)
4. Action Decoder génère la séquence d'actions MCP
5. Agentic Loop exécute chaque action via MCP
6. Observer capture l'état → feedback → replanification si nécessaire
7. Itération jusqu'à jeu complet
```

## Modalités

| Modalité | Entrée | Sortie | Usage |
|----------|--------|--------|-------|
| Texte | Description du jeu | Réponses, logs | Communication |
| Spatial | Voxels, positions | Meshes 3D | Construction |
| Code | Contexte du jeu | Scripts Luau | Gameplay |
| Audio | Description sonore | Audio PCM | SFX/Musique |
| Action | Plan d'étapes | Commandes MCP | Exécution |

## Stack

- **Langage**: Rust
- **ML**: candle (HuggingFace)
- **GPU**: WGPU (Vulkan → AMD/NVIDIA/Metal)
- **MCP**: HTTP/SSE vers Roblox Studio
- **Async**: Tokio

## Phases

1. **Phase 1** (actuelle): Fondation — MCP client, agentic loop, structure
2. **Phase 2**: Modèle — text encoder, fusion transformer
3. **Phase 3**: Modalités — 3D decoder, code decoder, audio
4. **Phase 4**: Entraînement — data pipeline, training loop
5. **Phase 5**: Intégration — end-to-end game creation
