import os
import json
import shutil
from pathlib import Path

def fix_lsa64_structure():
    base_dir = Path(__file__).resolve().parent / 'data' / 'processed' / 'lsa64'
    if not (base_dir / 'class_mapping.json').exists():
        print('class_mapping.json no encontrado.')
        return

    with open(base_dir / 'class_mapping.json') as f:
        class_to_idx = json.load(f)
    
    idx_to_class = {f"{v:03d}": k for k, v in class_to_idx.items()}

    moved_count = 0
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split
        seq_split_dir = base_dir / 'sequences' / split
        
        if not split_dir.exists():
            continue
            
        for npy in split_dir.glob('*.npy'):
            idx_str = npy.name[:3]
            class_name = idx_to_class.get(idx_str, f'class_{idx_str}')
            
            target_dir = seq_split_dir / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(npy), str(target_dir / npy.name))
            moved_count += 1
            
        # Eliminar carpeta original si quedó vacía
        try:
            split_dir.rmdir()
        except OSError:
            pass

    print(f'¡Estructura arreglada! {moved_count} archivos movidos a sequences/...')

if __name__ == '__main__':
    fix_lsa64_structure()
