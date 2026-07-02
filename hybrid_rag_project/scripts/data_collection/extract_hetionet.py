import os
import json
import bz2

def extract_neurology_subgraph():
    hetionet_file = "../hetionet/hetionet-v1.0.json.bz2"
    output_file = "../data/graphs/neurology_graph.json"

    print("1. Lettura del file Hetionet compresso...")
    try:
        with bz2.open(hetionet_file, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Errore: File non trovato in {hetionet_file}. Controlla il percorso.")
        return

    het_nodes = data['nodes']
    het_edges = data['edges']
    print(f"Dataset originale: {len(het_nodes)} nodi, {len(het_edges)} relazioni.")

    # 1. Trova i nodi radice (Alzheimer e Parkinson) in modo flessibile
    root_ids = set()
    
    for n in het_nodes:
        if n['kind'] == 'Disease':
            # Cerchiamo le parole chiave in minuscolo per evitare problemi di apostrofi o maiuscole
            name_lower = n['name'].lower()
            if 'alzheimer' in name_lower or 'parkinson' in name_lower:
                root_ids.add(n['identifier'])
                print(f" -> Trovata malattia: {n['name']} ({n['identifier']})")

    print(f"2. Nodi radice trovati: {root_ids}")

    # 2. Hop 1: Trova gli archi collegati alle malattie radice
    hop_1_edges = []
    hop_1_node_ids = set(root_ids)

    for e in het_edges:
        if e['source_id'][1] in root_ids or e['target_id'][1] in root_ids:
            hop_1_edges.append(e)
            hop_1_node_ids.add(e['source_id'][1])
            hop_1_node_ids.add(e['target_id'][1])

    # 3. Hop 2: Trova gli archi collegati ai farmaci (Compound) e Geni trovati al livello 1
    valid_hop_2_kinds = ['Side Effect', 'Pathway']
    hop_2_edges = []
    hop_2_node_ids = set()

    node_kinds = {n['identifier']: n['kind'] for n in het_nodes}

    for e in het_edges:
        src_id, tgt_id = e['source_id'][1], e['target_id'][1]
        
        if src_id in hop_1_node_ids and node_kinds.get(tgt_id) in valid_hop_2_kinds:
            hop_2_edges.append(e)
            hop_2_node_ids.add(tgt_id)
        elif tgt_id in hop_1_node_ids and node_kinds.get(src_id) in valid_hop_2_kinds:
            hop_2_edges.append(e)
            hop_2_node_ids.add(src_id)

    # 4. Uniamo tutto e assembliamo il JSON per il nostro RAG
    final_node_ids = hop_1_node_ids.union(hop_2_node_ids)
    
    out_nodes = []
    found_drugs = [] 

    for n in het_nodes:
        if n['identifier'] in final_node_ids:
            # FIX: Convertiamo sempre l'identifier in stringa prima di fare replace!
            clean_id = f"ns/{str(n['identifier']).replace(':', '_')}"
            out_nodes.append({
                "id": clean_id,
                "label": n['kind'],
                "properties": {
                    "title": n['name'],
                    "center": clean_id
                }
            })
            if n['kind'] == 'Compound':
                found_drugs.append(n['name'])

    out_edges = []
    for e in hop_1_edges + hop_2_edges:
        # FIX: Convertiamo in stringa anche gli ID sorgente e destinazione
        out_edges.append({
            "source": f"ns/{str(e['source_id'][1]).replace(':', '_')}",
            "target": f"ns/{str(e['target_id'][1]).replace(':', '_')}",
            "type": e['kind'].upper().replace(' ', '_')
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"nodes": out_nodes, "edges": out_edges}, f, indent=4)

    print(f"3. Sotto-grafo generato con successo in {output_file}!")
    print(f"-> Nodi finali: {len(out_nodes)}")
    print(f"-> Archi finali: {len(out_edges)}")
    print(f"\n[IMPORTANTE PER VECTOR/SQL] Ecco i farmaci trovati per cui dovremo scaricare paper e trial clinici:")
    for drug in found_drugs:
        print(f" - {drug}")

if __name__ == "__main__":
    extract_neurology_subgraph()