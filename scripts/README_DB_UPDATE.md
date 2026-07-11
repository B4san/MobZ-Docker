# Actualización de Base de Datos de Modelos Fireworks AI

## Resumen

Se han creado **tres bases de datos** como parte del proceso de actualización:

### 1. `mobz_model_profiles.json` (DB Original)
- **Total de modelos:** 294
- **Modelos con benchmarks:** 213
- **Estado:** No modificada (solo lectura)
- **Fuente:** Benchmarks de 2026 investigados manualmente

### 2. `mobz_model_profiles_new.json` (DB desde API)
- **Total de modelos:** 287
- **Modelos con benchmarks:** 1 (solo uno con datos de HF)
- **Fuente:** API de Fireworks AI + Hugging Face Open LLM Leaderboard
- **Estado:** Nueva, generada desde cero

### 3. `mobz_model_profiles_merged.json` (DB Fusionada) ⭐ **RECOMENDADA**
- **Total de modelos:** 287
- **Modelos con benchmarks:** 197
- **Fuente:** API de Fireworks AI + benchmarks preservados de DB original
- **Estado:** Fusionada, lista para usar

## Diferencias entre DBs

| Característica | Original | Nueva | Fusionada |
|---------------|----------|-------|-----------|
| Total modelos | 294 | 287 | 287 |
| Con benchmarks | 213 | 1 | 197 |
| Fuente datos | Manual | API+HF | API+Preservados |
| Estructura | ✅ Completa | ✅ Completa | ✅ Completa |

## Familias de Modelos (DB Fusionada)

| Familia | Cantidad |
|---------|----------|
| Qwen | 91 |
| Llama | 55 |
| Other | 23 |
| Deepseek | 22 |
| Gemma | 15 |
| Mistral | 15 |
| GLM | 10 |
| Nvidia | 10 |
| Kimi | 8 |
| Flux | 6 |

## Scripts Disponibles

### 1. `update_model_db.py`
Obtiene todos los modelos desde la API de Fireworks AI y crea una nueva DB.

```bash
python scripts/update_model_db.py
```

**Requiere:**
- `FIREWORKS_API_KEY` (opcional, si no está usa cuenta pública)
- Conexión a internet

**Genera:**
- `mobz_model_profiles_new.json`

### 2. `merge_databases.py`
Fusiona la DB original con la nueva, preservando benchmarks.

```bash
python scripts/merge_databases.py
```

**Requiere:**
- `mobz_model_profiles.json` (DB original)
- `mobz_model_profiles_new.json` (DB nueva)

**Genera:**
- `mobz_model_profiles_merged.json`

## Estructura de Cada Perfil

```json
{
  "model_id": "accounts/fireworks/models/deepseek-v4-pro",
  "display_name": "DeepSeek-V4-Pro",
  "provider": "fireworks",
  "family": "Deepseek",
  "parameters": "Dense",
  "context_length": 131072,
  "release_date": "2024-2025",
  "cost": {
    "input_per_million": 0.435,
    "output_per_million": 0.87,
    "cached_input_per_million": null
  },
  "raw_benchmarks": {
    "mmlu": 0.9,
    "mmlu_pro": 0.855,
    "gpqa_diamond": 0.901,
    "humaneval": 0.93,
    "gsm8k": 0.97,
    "math_500": 0.97,
    "ifeval": 0.88,
    "bbh": 0.9,
    "livecodebench": 0.935
  },
  "composite_indices": {
    "knowledge_index": 0.8853,
    "instruction_index": 0.88,
    "coding_index": 0.9325,
    "reasoning_index": 0.9237,
    "math_index": 0.97
  },
  "cognitive_profile": {
    "reasoning_depth": "excellent",
    "instruction_following": "unknown",
    "json_reliability": "fair",
    "long_context_handling": "good",
    "hallucination_tendency": "medium",
    "tool_usage": "good",
    "verbosity": "medium",
    "consistency": "medium",
    "multilingual": "good",
    "creative_writing": "good"
  },
  "strengths": ["coding", "reasoning"],
  "weaknesses": [],
  "performance_by_difficulty": {
    "easy": { "estimated_accuracy": 0.968, "recommended": true, "reason": "suitable" },
    "medium": { "estimated_accuracy": 0.866, "recommended": true, "reason": "balanced" },
    "hard": { "estimated_accuracy": 0.855, "recommended": true, "reason": "strong" }
  },
  "routing_tags": ["deepseek", "general"],
  "benchmark_sources": "Enriched from published 2026 benchmarks/leaderboards",
  "capability_scores": {
    "knowledge": 0.8853,
    "math": 0.97,
    "coding": 0.9325,
    "reasoning": 0.9237,
    "instruction_following": 0.88,
    "summary": 0.8826,
    "ner": 0.9184,
    "sentiment": 0.9808
  },
  "json_reliability_score": 0.8,
  "verbosity_factor": 1.0,
  "estimated_output_tokens": 220
}
```

## Recomendación

Usar `mobz_model_profiles_merged.json` como la base de datos principal, ya que:
1. Contiene todos los modelos actuales de Fireworks AI (287)
2. Preserva los benchmarks investigados de la DB original (197 modelos)
3. Mantiene la estructura completa compatible con MobZ
4. Incluye información de costos y capacidades

## Próximos Pasos

Para mejorar la DB fusionada:

1. **Agregar benchmarks faltantes:** Los 90 modelos sin benchmarks podrían obtenerse de:
   - Hugging Face Open LLM Leaderboard (con mejor mapeo de nombres)
   - Model cards oficiales
   - Papers de los modelos

2. **Actualizar costos:** Verificar costos actuales desde la API de Fireworks

3. **Validar routing:** Probar que el policy engine funciona correctamente con la nueva DB

4. **Agregar modelos nuevos:** Los 22 modelos nuevos podrían requerir benchmarks manuales

## Notas Técnicas

- La API de Fireworks devuelve hasta 200 modelos por página
- El dataset de HF `open-llm-leaderboard/results` tiene problemas de carga, se usó `lucyknada/mmlu-leaderboard` como alternativa
- La normalización de IDs es necesaria porque la API usa `accounts/fireworks/models/...` mientras que la DB original usa `fireworks/...`
