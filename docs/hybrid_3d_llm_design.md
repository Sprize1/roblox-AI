# Hybrid 3D-LLM — Design (backbone LFM2.5-8B-A1B + modalité géométrie)

But : générer une **place Roblox complète et cohérente** (parts 3D + scripts Luau couplés) à partir
d'une description. Approche **hybride** : backbone LLM pré-entraîné (langage/code/raisonnement, on ne le
réapprend pas) + une **modalité géométrie greffée et entraînée from-scratch**, le tout dans une seule
séquence auto-régressive. Le LLM apporte les scripts, la modalité apporte la 3D, l'attention partagée
apporte le couplage.

Statut : spec V1. Rien n'est codé. Décisions ouvertes listées en §10.

**Décisions verrouillées**
- Backbone = **LFM2.5-8B-A1B** (choisi pour efficacité MoE + récence ; Qwen2.5-Coder gardé en fallback si le couplage long-range souffre, cf §10.1).
- Publication = **LFM Open License v1.0 + rider attribution** (hérité, non négociable : cap commercial 10 M$ + crédit Liquid). Le **code maison** (geo_encoder, têtes, serializer, training) reste licenciable librement (Apache/MIT au choix).

---

## 1. Backbone : LFM2.5-8B-A1B (faits du config.json)

| Champ | Valeur | Impact design |
|---|---|---|
| `model_type` / classe | `lfm2_moe` / `Lfm2MoeForCausalLM` | code modeling custom (dans Transformers ≥5.9) |
| `hidden_size` | **2048** | dim de sortie de l'encodeur géo + dim d'entrée des têtes |
| `layer_types` | 18 `conv` + 6 `full_attention` (idx 2,6,10,14,18,21) | le **long-range spatial repose sur 6 couches d'attention** + RoPE |
| `num_attention_heads` / `kv` | 32 / 8 (GQA, head_dim 64) | — |
| MoE | 32 experts, 4 actifs, `moe_intermediate_size`=1792, 2 couches denses | 1.5B actifs → train/serve efficaces |
| `vocab_size` | 128000 | tokens texte + tokens structurels réutilisent ce vocab |
| `tie_word_embeddings` | true | tête texte = embeddingᵀ (réutilisée telle quelle) |
| `rope_theta` / ctx | 5e6 / 128000 | contexte 128k → une place compacte tient en entier |
| `conv_L_cache` | 3 | conv courte (local) → mixing global = via les 6 couches attn |

Variante à fine-tuner : **`LFM2.5-8B-A1B-Base`** (pas l'instruct).
Licence : LFM Open License v1.0 → commercial OK si CA < 10 M$/an ; Derivative Works redistribuables.

---

## 2. Schéma de séquence mixte

Une place = **un document auto-régressif** mélangeant 4 modalités de tokens :

- `STRUCT` — tokens structurels : `[PLACE] [GENRE=x] [PLAN] [MODEL] [/MODEL] [GRID] [ROW] [SCRIPT] [/SCRIPT] [/PLACE]` …
  → réutilisent des IDs réservés du vocab 128k (tokens spéciaux ajoutés au tokenizer).
- `GEO` — **un token = une part**, embedding = projection de ses propriétés 3D continues (§3).
- `REF` — pointeur vers une part déjà émise (attache de script, référence inter-part).
- `TEXT` — Luau verbatim + texte libre (tokenizer du backbone).

### Ordre de sérialisation (= curriculum de génération, coarse-to-fine)
```
[PLACE][GENRE=obby]
[PLAN] short spec: objectifs, nb rooms, compte de parts/scripts   (TEXT court)
[MODEL id=0 "Spawn"]                                              (STRUCT + GEO d'ancrage)
   <GEO part id=1 ...>  <GEO part id=2 ...>
   [GRID ...]  (op d'instancing, lossless, optionnel)             (STRUCT + params)
[/MODEL]
[MODEL id=3 "Course"]
   <GEO part id=4 ...> ...
[SCRIPT class=Script attach=REF(4)]                               (STRUCT + REF)
```luau
script.Parent.Touched:Connect(function(hit) ... end)             (TEXT)
```
[/SCRIPT]
[/PLACE]
```
Plan d'abord → le modèle s'engage sur la structure. Géo (hiérarchie Models → parts) ensuite. Scripts
en dernier → ils peuvent référencer n'importe quelle part par `id`.

---

## 3. Entrée : chaque modalité → vecteur 2048

- **TEXT / STRUCT** : `embed_tokens` du backbone (existant, tied). Les tokens STRUCT sont des specials ajoutés.
- **GEO (part token)** : `geo_encoder(features) → R^2048`, un petit MLP (ex. 2 couches, GELU).
  `features` = concat de :
  - position **relative au parent Model**, quantifiée puis embeddée OU passée en continu normalisé
  - taille (x,y,z)
  - rotation en **6D** (2 colonnes de la matrice — continu, sans discontinuité)
  - forme : embedding(shape_enum)
  - couleur : embedding(BrickColor) ou RGB normalisé
  - matériau : embedding(material_enum)
  - flags : anchored / cancollide (bits)
- **REF** : embedding du token-référence **lié à la représentation de la part pointée** (copie de son
  hidden/embedding) → le script « porte » la géo de la part.
- **Modality tag** : on **ajoute** à chaque embedding un `modality_embedding[type]` (4 types) pour que le
  backbone sache quelle modalité il lit.

Tous les vecteurs sont fournis via `inputs_embeds` (injection au niveau embedding → **agnostique à
l'archi conv/attn** du backbone).

---

## 4. Quantification (têtes de sortie) — fin = quasi sans perte

Choix adaptés à Roblox (data réelle) :
- **Position** (relative parent) : par axe, B bins (ex. 1024) sur une plage clampée (ex. ±256 studs),
  résolution ~0.5 stud. Origine de chaque Model = tête « absolue » plus grossière séparée.
- **Taille** : bins **log-espacés** (ex. 256) — les tailles vont du petit au géant.
- **Rotation** : hybride — tête « snap » sur rotations courantes ({0,90,180,270}³, couvre la majorité
  des builds) **+** résidu fin optionnel. (Évite de gâcher des bins sur du continu rarement utilisé.)
- **Forme** : classif sur enum {Block, Ball, Cylinder, Wedge, CornerWedge, Truss, MeshPart, …}.
- **Couleur** : classif sur la **palette BrickColor** (~128) — colle à la data, compact.
- **Matériau** : classif sur enum Material (~20).

→ La géo devient surtout de la **classification multi-tête** (apprenable, multimodal-friendly), pas de
la régression molle.

---

## 5. Sortie : tête de type + têtes par modalité

À chaque position, sur le hidden 2048 :
1. **Type head** (4 voies) prédit la modalité du **prochain** token.
2. Tête sélectionnée :
   - `TEXT` → **tête LM tied existante** (softmax 128k). Zéro coût, réutilise le backbone.
   - `GEO` → les têtes de classif de §4 (pos x/y/z, size x/y/z, rot, shape, color, material, flags).
   - `REF` → **pointer** : score = dot(hidden_courant, hidden_des_parts_émises), softmax **masqué aux
     ids existants** → garantit zéro référence pendante.
   - `STRUCT` → petit softmax sur le sous-vocab structurel.

---

## 6. Perte

`L = w_type·CE_type + w_text·CE_text + w_geo·Σ CE_attr + w_ref·CE_ref + w_struct·CE_struct`
- chaque terme **masqué** par la vraie modalité de la position.
- équilibrage : normaliser par le nb de tokens de chaque modalité ; démarrer `w_geo` un peu haut (la
  géo est le neuf). Surveiller que CE_text ne s'effondre pas (sinon le LLM « oublie » le Luau).

---

## 7. Recette d'entraînement (spécifique LFM2.5)

Partir de **LFM2.5-8B-A1B-Base**.

**Phase A — alignement de modalité (backbone gelé).**
Entraîner uniquement les **nouveaux modules** : `geo_encoder`, têtes géo, type head, ref head,
`modality_embedding`, embeddings des tokens STRUCT. Objectif : projeter la géo dans l'espace du LLM
sans casser ses poids.

**Phase B — fine-tune joint (LoRA sur backbone).**
Ajouter du **LoRA** ciblant en priorité les **6 couches `full_attention`** (`q_proj,k_proj,v_proj,o_proj`)
+ éventuellement les projections d'experts MoE. Modules nouveaux restent en full-train.
> ⚠️ Confirmer les **noms exacts des modules** via `model.named_modules()` (archi `lfm2_moe` custom :
> les couches conv et le routeur MoE ont des noms propres). Ne pas supposer les cibles LoRA.

Hyperparams de départ : LoRA r=16–32, lr_backbone≈1e-4 (LoRA), lr_modules≈3e-4, bf16,
gradient checkpointing, cosine + warmup. Contexte d'entraînement : longueur d'une place compacte
(viser ≤ 16k au début ; 128k dispo si besoin).

**Faisabilité H200 (141 Go).** 8.3B total / 1.5B actifs + LoRA + modules légers → **large**. Full-FT du
backbone possible aussi mais inutile : LoRA suffit et garde le prior langage.

**Données.** Réécrire `scripts/build_structured_v2.py` pour émettre des **records structurés**
(parts avec rotation/forme/hiérarchie + scripts + attaches par id), PAS le flat-text actuel. Un
*collator* mappe chaque record vers le flux de tokens mixte (§2). Source : les 2092 `.rbxl`.

---

## 8. Inférence & cohérence

- **Coarse-to-fine** : décoder plan → géo → scripts (l'ordre d'entraînement).
- **Décodage contraint** : la tête REF est masquée aux ids inexistants → **aucune ref pendante**.
  On peut aussi contraindre la syntaxe STRUCT (grammaire) et borner les bins géo.
- **Contexte** : une place re-sérialisée compacte devrait tenir dans 128k → génération **place entière
  sans sliding-window** (gros avantage vs le plan chunking). Fallback fenêtre glissante si dépassement.
- **Expander** : un script déterministe transforme la sortie (tokens géo + ops d'instancing + scripts)
  en place Roblox (via le MCP Studio).

---

## 9. Ordre de construction (MVP → complet)

- **V1 (MVP)** : modalité GEO (parts) + TEXT (scripts) + REF d'**attache** uniquement.
  Pas d'instancing, pas de normalisation des refs internes au Luau. Hiérarchie Model basique.
  → valide la greffe + le couplage de base.
- **V2** : ops d'**instancing** (GRID/ROW/MIRROR, lossless) pour compresser les builds répétitifs ;
  **normalisation des refs Luau** (`script.Parent`, `WaitForChild("X")` → `REF(id)`) ; rotations fines.
- **V3** : plan/level-design plus riche, contraintes de jouabilité, RL sur cohérence.

---

## 10. Risques / décisions ouvertes

1. **Backbone conv-lourd** : le long-range spatial repose sur 6 couches d'attention seulement.
   À surveiller — si le couplage géo↔script souffre, augmenter le LoRA sur ces couches ou tester un
   backbone full-attention (Qwen2.5-Coder) en comparaison.
2. **Échelle data** : ~2300 places. Suffisant pour la modalité + couplage ? Sinon : augmentation
   (perturbations géo), et pré-chauffer le côté script avec `Roblox/luau_corpus` (vérifier sa licence).
3. **Résolution de quantif** vs budget tokens (1 part = 1 token en entrée, mais N attributs en sortie).
4. **Équilibrage de perte** géo vs texte (risque d'oubli du Luau).
5. **Licences en cascade** : LFM (<10 M$), luau_corpus (?), et droits des `.rbxl` uncopylocked
   (redistribuer un modèle entraîné dessus ≠ forcément permis).
6. **Recherche vs produit** : si recherche d'archi pure → from-scratch reste une option assumée ;
   ce design vise le **produit** (backbone permissif + neuf greffé).
