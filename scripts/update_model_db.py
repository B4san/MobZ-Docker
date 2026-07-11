#!/usr/bin/env python3
"""
Script para actualizar la base de datos de modelos de Fireworks AI.
1. Obtiene todos los modelos desde la API de Fireworks
2. Descarga benchmarks desde Hugging Face Open LLM Leaderboard
3. Crea una nueva DB no relacional con toda la información
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
from datasets import load_dataset
import time


class FireworksModelUpdater:
    """Actualiza la DB de modelos de Fireworks AI con benchmarks de HF."""
    
    def __init__(self, api_key: Optional[str] = None, account_id: str = "fireworks"):
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        self.account_id = account_id
        self.base_url = "https://api.fireworks.ai/v1"
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.existing_profiles = {}
    
    def load_existing_profiles(self, db_path: str = "mobz_model_profiles.json") -> Dict[str, Dict]:
        """Carga la DB existente para preservar benchmarks."""
        script_dir = os.path.dirname(__file__)
        db_file = os.path.join(script_dir, "..", db_path)
        
        if os.path.exists(db_file):
            try:
                with open(db_file, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                
                # Crear diccionario indexado por model_id
                profile_dict = {}
                for p in profiles:
                    model_id = p.get("model_id", "")
                    if model_id:
                        profile_dict[model_id] = p
                
                print(f"   ✓ DB existente cargada: {len(profile_dict)} perfiles")
                return profile_dict
            except Exception as e:
                print(f"   ⚠ Error cargando DB existente: {e}")
        
        return {}
        
    def fetch_all_fireworks_models(self) -> List[Dict[str, Any]]:
        """Obtiene TODOS los modelos disponibles en Fireworks AI."""
        all_models = []
        page_token = None
        page_size = 200  # Máximo permitido por la API
        
        print("📥 Obteniendo modelos de Fireworks AI...")
        
        while True:
            url = f"{self.base_url}/accounts/{self.account_id}/models"
            params = {"pageSize": page_size}
            
            if page_token:
                params["pageToken"] = page_token
            
            try:
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                models = data.get("models", [])
                all_models.extend(models)
                
                print(f"   ✓ Obtenidos {len(models)} modelos (total: {len(all_models)})")
                
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"   ✗ Error obteniendo modelos: {e}")
                break
        
        print(f"\n✅ Total de modelos obtenidos: {len(all_models)}")
        return all_models
    
    def load_hf_leaderboard(self) -> pd.DataFrame:
        """Descarga el dataset de Hugging Face Open LLM Leaderboard."""
        print("\n📊 Descargando Hugging Face Open LLM Leaderboard...")
        
        # Intentar varios datasets
        datasets_to_try = [
            ("open-llm-leaderboard/results", {}),
            ("lucyknada/mmlu-leaderboard", {}),
        ]
        
        for dataset_name, kwargs in datasets_to_try:
            try:
                print(f"   Intentando: {dataset_name}...")
                dataset = load_dataset(dataset_name, **kwargs)
                df = dataset["train"].to_pandas()
                print(f"   ✓ Dataset cargado: {len(df)} entradas")
                return df
            except Exception as e:
                print(f"   ✗ Error con {dataset_name}: {str(e)[:50]}...")
                continue
        
        print("   ⚠ No se pudo cargar ningún dataset. Usando DB existente como referencia.")
        return pd.DataFrame()
    
    def extract_model_family(self, model_name: str) -> str:
        """Extrae la familia del modelo desde su nombre."""
        name_lower = model_name.lower()
        
        families = {
            "qwen": ["qwen"],
            "deepseek": ["deepseek"],
            "llama": ["llama"],
            "glm": ["glm"],
            "gemma": ["gemma"],
            "kimi": ["kimi"],
            "minimax": ["minimax"],
            "mistral": ["mistral", "ministral"],
            "nvidia": ["nemotron"],
            "gpt": ["gpt-oss"],
            "cogito": ["cogito"],
            "devstral": ["devstral"],
            "internvl": ["internvl"],
            "molmo": ["molmo"],
            "voyage": ["voyage"],
            "ernie": ["ernie"],
            "seed": ["seed"],
            "kat": ["kat"],
            "dobby": ["dobby"],
            "step": ["step"],
            "fare": ["fare"],
            "mirothinker": ["mirothinker"],
            "paddleocr": ["paddleocr"],
            "rolm": ["rolm"],
            "flux": ["flux"],
        }
        
        for family, keywords in families.items():
            if any(kw in name_lower for kw in keywords):
                return family.upper() if family in ["gpt", "glm"] else family.capitalize()
        
        return "Other"
    
    def extract_display_name(self, model_id: str) -> str:
        """Extrae un nombre legible del model_id."""
        name = model_id.replace("accounts/fireworks/models/", "")
        name = name.replace("fireworks/", "")
        name = name.replace("deepseek-ai/", "")
        name = name.replace("cogito/", "")
        name = name.replace("sentientfoundation-serverless/", "")
        
        name = name.replace("-", " ").replace("_", " ")
        name = " ".join(word.capitalize() for word in name.split())
        
        return name
    
    def map_fireworks_to_hf_name(self, fw_model_id: str) -> List[str]:
        """Mapea el ID de modelo de Fireworks a posibles nombres en HF."""
        base_name = fw_model_id.replace("accounts/fireworks/models/", "")
        base_name = base_name.replace("fireworks/", "")
        
        variations = []
        
        if "qwen" in base_name:
            variations.extend([
                f"Qwen/Qwen-{base_name.split('qwen')[-1]}",
                f"Qwen/Qwen{base_name.split('qwen')[-1]}",
                f"Qwen/Qwen2.5-{base_name.split('qwen')[-1]}",
            ])
        elif "deepseek" in base_name:
            variations.extend([
                f"deepseek-ai/{base_name}",
                f"deepseek-ai/DeepSeek-{base_name.split('deepseek')[-1]}",
            ])
        elif "llama" in base_name:
            variations.extend([
                f"meta-llama/{base_name}",
                f"meta-llama/Llama-{base_name.split('llama')[-1]}",
            ])
        
        return variations
    
    def find_benchmarks_in_hf(self, model_id: str, hf_df: pd.DataFrame, existing_profiles: Dict[str, Dict] = None) -> Dict[str, Any]:
        """Busca benchmarks del modelo en el dataset de HF o en la DB existente."""
        benchmarks = {}
        
        # Primero, intentar con DB existente si está disponible
        if existing_profiles and model_id in existing_profiles:
            existing = existing_profiles[model_id]
            if "raw_benchmarks" in existing:
                benchmarks = existing["raw_benchmarks"].copy()
                return benchmarks
        
        # Si no hay en DB existente, buscar en HF
        if not hf_df.empty:
            search_names = self.map_fireworks_to_hf_name(model_id)
            search_names.append(model_id.split("/")[-1])
            
            for search_name in search_names:
                try:
                    matches = hf_df[hf_df['model'].str.lower().str.contains(search_name.lower(), na=False)]
                    
                    if not matches.empty:
                        row = matches.iloc[0]
                        
                        benchmark_cols = [
                            'mmlu', 'mmlu_pro', 'gpqa_diamond', 'humaneval', 
                            'gsm8k', 'math_500', 'ifeval', 'bbh', 'livecodebench',
                            'arena_hard'
                        ]
                        
                        for col in benchmark_cols:
                            if col in row and pd.notna(row[col]):
                                try:
                                    benchmarks[col] = float(row[col])
                                except:
                                    pass
                        
                        if benchmarks:
                            break
                except:
                    pass
        
        return benchmarks
    
    def compute_composite_indices(self, benchmarks: Dict[str, Any]) -> Dict[str, Optional[float]]:
        """Calcula índices compuestos a partir de benchmarks crudos."""
        indices = {
            "knowledge_index": None,
            "instruction_index": None,
            "coding_index": None,
            "reasoning_index": None,
            "math_index": None,
        }
        
        if not benchmarks:
            return indices
        
        knowledge_scores = [v for k, v in benchmarks.items() if k in ['mmlu', 'mmlu_pro', 'gpqa_diamond'] and v is not None]
        if knowledge_scores:
            indices["knowledge_index"] = sum(knowledge_scores) / len(knowledge_scores)
        
        if 'ifeval' in benchmarks and benchmarks['ifeval'] is not None:
            indices["instruction_index"] = benchmarks['ifeval']
        
        coding_scores = [v for k, v in benchmarks.items() if k in ['humaneval', 'livecodebench'] and v is not None]
        if coding_scores:
            indices["coding_index"] = sum(coding_scores) / len(coding_scores)
        
        reasoning_scores = [v for k, v in benchmarks.items() if k in ['gpqa_diamond', 'bbh'] and v is not None]
        if reasoning_scores:
            indices["reasoning_index"] = sum(reasoning_scores) / len(reasoning_scores)
        
        math_scores = [v for k, v in benchmarks.items() if k in ['gsm8k', 'math_500'] and v is not None]
        if math_scores:
            indices["math_index"] = sum(math_scores) / len(math_scores)
        
        return indices
    
    def compute_capability_scores(self, benchmarks: Dict[str, Any], composite: Dict[str, Optional[float]]) -> Dict[str, float]:
        """Calcula scores de capacidad para routing."""
        capabilities = {}
        
        if composite.get("knowledge_index"):
            capabilities["knowledge"] = composite["knowledge_index"]
        else:
            capabilities["knowledge"] = 0.72
        
        if composite.get("math_index"):
            capabilities["math"] = composite["math_index"]
        else:
            capabilities["math"] = 0.72
        
        if composite.get("coding_index"):
            capabilities["coding"] = composite["coding_index"]
        else:
            capabilities["coding"] = 0.82
        
        if composite.get("reasoning_index"):
            capabilities["reasoning"] = composite["reasoning_index"]
        else:
            capabilities["reasoning"] = 0.72
        
        if composite.get("instruction_index"):
            capabilities["instruction_following"] = composite["instruction_index"]
        else:
            capabilities["instruction_following"] = 0.72
        
        summary_scores = [
            capabilities["knowledge"],
            capabilities["reasoning"],
            capabilities.get("instruction_following", 0.72)
        ]
        capabilities["summary"] = sum(summary_scores) / len(summary_scores)
        
        capabilities["ner"] = min(capabilities["knowledge"] + 0.06, 1.0)
        capabilities["sentiment"] = 0.982
        
        return capabilities
    
    def estimate_performance_by_difficulty(self, composite: Dict[str, Optional[float]]) -> Dict[str, Dict]:
        """Estima el rendimiento por nivel de dificultad."""
        avg_score = sum(v for v in composite.values() if v is not None) / max(1, sum(1 for v in composite.values() if v is not None))
        
        performance = {
            "easy": {
                "estimated_accuracy": min(avg_score + 0.15, 0.99),
                "recommended": True,
                "reason": "suitable"
            },
            "medium": {
                "estimated_accuracy": min(avg_score + 0.05, 0.95),
                "recommended": True,
                "reason": "balanced"
            },
            "hard": {
                "estimated_accuracy": max(avg_score - 0.05, 0.65),
                "recommended": avg_score >= 0.75,
                "reason": "strong" if avg_score >= 0.75 else "unreliable"
            }
        }
        
        if performance["easy"]["estimated_accuracy"] > 0.97:
            performance["easy"]["recommended"] = False
            performance["easy"]["reason"] = "overkill"
        
        return performance
    
    def build_model_profile(self, fw_model: Dict, hf_benchmarks: Dict) -> Dict[str, Any]:
        """Construye el perfil completo de un modelo."""
        model_id = fw_model.get("name", "")
        display_name = fw_model.get("displayName", self.extract_display_name(model_id))
        
        context_length = fw_model.get("contextLength", 131072)
        
        composite = self.compute_composite_indices(hf_benchmarks)
        capabilities = self.compute_capability_scores(hf_benchmarks, composite)
        performance = self.estimate_performance_by_difficulty(composite)
        
        profile = {
            "model_id": model_id,
            "display_name": display_name,
            "provider": "fireworks" if "fireworks" in model_id else model_id.split("/")[0],
            "family": self.extract_model_family(model_id),
            "parameters": "Dense",
            "context_length": context_length,
            "release_date": "2024-2025",
            "cost": {
                "input_per_million": 0.5,
                "output_per_million": 2.0,
                "cached_input_per_million": None
            },
            "raw_benchmarks": hf_benchmarks,
            "composite_indices": composite,
            "cognitive_profile": {
                "reasoning_depth": "excellent" if (composite.get("reasoning_index") or 0) >= 0.85 else "good" if (composite.get("reasoning_index") or 0) >= 0.75 else "fair",
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
            "strengths": [],
            "weaknesses": [],
            "performance_by_difficulty": performance,
            "routing_tags": [self.extract_model_family(model_id).lower(), "general"],
            "benchmark_sources": "Hugging Face Open LLM Leaderboard" if hf_benchmarks else "Default estimates",
            "capability_scores": capabilities,
            "json_reliability_score": 0.8,
            "verbosity_factor": 1.0,
            "estimated_output_tokens": 220
        }
        
        if capabilities["coding"] >= 0.8:
            profile["strengths"].append("coding")
        if capabilities["math"] >= 0.85:
            profile["strengths"].append("math")
        if capabilities["reasoning"] >= 0.85:
            profile["strengths"].append("reasoning")
        if capabilities["knowledge"] >= 0.85:
            profile["strengths"].append("knowledge")
        
        if capabilities["reasoning"] < 0.7:
            profile["weaknesses"].append("complex_reasoning")
        
        return profile
    
    def update_database(self, output_path: str = "mobz_model_profiles_new.json"):
        """Ejecuta el proceso completo de actualización."""
        print("=" * 60)
        print("🔄 ACTUALIZACIÓN DE BASE DE DATOS DE MODELOS FIREWORKS AI")
        print("=" * 60)
        
        # Cargar DB existente para preservar benchmarks
        print("\n📚 Cargando base de datos existente...")
        self.existing_profiles = self.load_existing_profiles()
        
        fw_models = self.fetch_all_fireworks_models()
        
        if not fw_models:
            print("❌ No se obtuvieron modelos de Fireworks. Abortando.")
            return
        
        hf_df = self.load_hf_leaderboard()
        
        print("\n🔧 Construyendo perfiles de modelos...")
        profiles = []
        
        for i, fw_model in enumerate(fw_models, 1):
            model_id = fw_model.get("name", "")
            hf_benchmarks = self.find_benchmarks_in_hf(model_id, hf_df, self.existing_profiles)
            profile = self.build_model_profile(fw_model, hf_benchmarks)
            profiles.append(profile)
            
            if i % 50 == 0:
                print(f"   Procesados {i}/{len(fw_models)} modelos...")
        
        print(f"\n✅ Perfiles construidos: {len(profiles)}")
        
        output_file = os.path.join(os.path.dirname(__file__), "..", output_path)
        
        print(f"\n💾 Guardando nueva base de datos en: {output_file}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "=" * 60)
        print(f"✨ ACTUALIZACIÓN COMPLETADA")
        print(f"   - Modelos procesados: {len(profiles)}")
        print(f"   - Archivo guardado: {output_file}")
        print(f"   - Modelos con benchmarks: {sum(1 for p in profiles if p['raw_benchmarks'])}")
        print(f"=" * 60)
        
        return profiles


def main():
    """Función principal."""
    api_key = os.getenv("FIREWORKS_API_KEY")
    
    if not api_key:
        print("⚠️  FIREWORKS_API_KEY no está configurada.")
        print("   Configúrala con: export FIREWORKS_API_KEY='tu-api-key'")
        print("   Continuando sin API key (usando cuenta pública)...")
    
    updater = FireworksModelUpdater(api_key=api_key)
    updater.update_database()


if __name__ == "__main__":
    main()
