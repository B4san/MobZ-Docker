#!/usr/bin/env python3
"""
Script para fusionar la DB antigua con la nueva, preservando benchmarks.
"""

import json
import os
from typing import Dict, List
import re


def normalize_model_id(model_id: str) -> str:
    """Normaliza el ID del modelo para comparación."""
    # Remover prefijos comunes
    normalized = model_id.lower()
    normalized = normalized.replace('accounts/fireworks/models/', '')
    normalized = normalized.replace('fireworks/', '')
    normalized = normalized.replace('deepseek-ai/', '')
    normalized = normalized.replace('cogito/', '')
    normalized = normalized.replace('sentientfoundation-serverless/', '')
    
    # Remover guiones y puntos para comparación
    normalized = re.sub(r'[-._]', '', normalized)
    
    return normalized


def merge_databases(old_db_path: str, new_db_path: str, output_path: str):
    """Fusiona las dos bases de datos, preservando benchmarks de la antigua."""
    
    print("=" * 70)
    print("🔄 FUSIÓN DE BASES DE DATOS")
    print("=" * 70)
    
    # Cargar ambas DBs
    with open(old_db_path, 'r') as f:
        old_db = json.load(f)
    
    with open(new_db_path, 'r') as f:
        new_db = json.load(f)
    
    print(f"\n📚 DB antigua: {len(old_db)} modelos")
    print(f"📚 DB nueva: {len(new_db)} modelos")
    
    # Crear índice normalizado de la DB antigua
    old_index: Dict[str, Dict] = {}
    for profile in old_db:
        model_id = profile.get('model_id', '')
        if model_id:
            normalized = normalize_model_id(model_id)
            old_index[normalized] = profile
    
    # Fusionar: usar datos de la DB nueva, pero preservar benchmarks de la antigua
    merged_db: List[Dict] = []
    preserved_count = 0
    
    for new_profile in new_db:
        model_id = new_profile.get('model_id', '')
        normalized = normalize_model_id(model_id)
        
        if normalized in old_index:
            # Preservar benchmarks y datos enriquecidos de la DB antigua
            old_profile = old_index[normalized]
            
            # Crear perfil fusionado
            merged_profile = new_profile.copy()
            
            # Preservar benchmarks si existen en la DB antigua
            if old_profile.get('raw_benchmarks'):
                merged_profile['raw_benchmarks'] = old_profile['raw_benchmarks']
                merged_profile['composite_indices'] = old_profile.get('composite_indices', {})
                merged_profile['capability_scores'] = old_profile.get('capability_scores', {})
                merged_profile['benchmark_sources'] = old_profile.get('benchmark_sources', 'Preserved from old DB')
                preserved_count += 1
            
            # Preservar costos si existen
            if 'cost' in old_profile and old_profile['cost'].get('input_per_million') not in [None, 'N/A', 0]:
                merged_profile['cost'] = old_profile['cost']
            
            # Preservar otros campos enriquecidos
            for field in ['strengths', 'weaknesses', 'cognitive_profile', 'performance_by_difficulty']:
                if field in old_profile:
                    merged_profile[field] = old_profile[field]
            
            merged_db.append(merged_profile)
        else:
            # Modelo nuevo, usar datos de la DB nueva
            merged_db.append(new_profile)
    
    print(f"\n✅ Benchmarks preservados: {preserved_count}")
    print(f"✅ Modelos en DB fusionada: {len(merged_db)}")
    
    # Guardar DB fusionada
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_db, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Archivo guardado: {output_path}")
    print("=" * 70)
    
    return merged_db


def main():
    """Función principal."""
    base_dir = os.path.dirname(__file__)
    
    old_db = os.path.join(base_dir, '..', 'mobz_model_profiles.json')
    new_db = os.path.join(base_dir, '..', 'mobz_model_profiles_new.json')
    output_db = os.path.join(base_dir, '..', 'mobz_model_profiles_merged.json')
    
    merge_databases(old_db, new_db, output_db)


if __name__ == "__main__":
    main()
