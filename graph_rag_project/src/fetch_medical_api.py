import os
import json
import requests
import time

def fetch_trials_to_json():
    # Definisce il percorso di output, assicurandosi che la cartella esista
    output_file = "../data/tables/clinical_trials_data.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # I 32 farmaci esatti ricavati dallo script del Grafo
    target_drugs = [
        "Apomorphine", "Benzatropine", "Trihexyphenidyl", "Atropine", "Ropinirole", 
        "Gabapentin", "Fludrocortisone", "Rasagiline", "Memantine", "Bromocriptine", 
        "Entacapone", "Clonazepam", "Rivastigmine", "Galantamine", "Diphenhydramine", 
        "Carbidopa", "Pergolide", "Amantadine", "Cabergoline", "Modafinil", "L-DOPA", 
        "Rotigotine", "Procyclidine", "Donepezil", "Ramelteon", "Pramipexole", 
        "Biperiden", "Haloperidol", "Tolcapone", "Hyoscyamine", "Quetiapine", "Selegiline"
    ]

    base_url = "https://clinicaltrials.gov/api/v2/studies"
    all_studies = []

    print(f"Inizio download dei trial clinici per {len(target_drugs)} farmaci...")

    for drug_name in target_drugs:
        print(f"\n[API] Cerco studi per: {drug_name}...")
        
        # Parametri API: cerchiamo per intervento (intr)
        params = {
            "query.intr": drug_name,
            "pageSize": 100  # Paginazione a 100 per richiesta
        }
        
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            studies_list = data.get('studies', [])
            
            print(f" -> Trovati {len(studies_list)} studi. Estrazione dei campi rilevanti...")
            
            for study_data in studies_list:
                protocol = study_data.get('protocolSection', {})
                id_module = protocol.get('identificationModule', {})
                status_module = protocol.get('statusModule', {})
                design_module = protocol.get('designModule', {})
                sponsor_module = protocol.get('sponsorCollaboratorsModule', {})
                contacts_module = protocol.get('contactsLocationsModule', {})
                
                nct_id = id_module.get('nctId')
                if not nct_id:
                    continue

                # Estrazione sicura della Fase (es. Phase 2, Phase 3)
                phases = design_module.get('phases', [])
                phase_str = ", ".join(phases) if phases else "Not Specified"
                
                # Estrazione Sponsor Principale
                lead_sponsor = sponsor_module.get('leadSponsor', {}).get('name', 'Unknown')
                
                # Estrazione Locazioni (Ospedali/Cliniche)
                locations = []
                for loc in contacts_module.get('locations', []):
                    locations.append({
                        "facility": loc.get('facility', 'Unknown'),
                        "city": loc.get('city', 'Unknown'),
                        "country": loc.get('country', 'Unknown')
                    })

                # Creazione record normalizzato in formato dizionario
                study_obj = {
                    "drug_name": drug_name,
                    "nct_id": nct_id,
                    "title": id_module.get('officialTitle', id_module.get('briefTitle', 'Unknown')),
                    "status": status_module.get('overallStatus', 'Unknown'),
                    "phase": phase_str,
                    "enrollment": design_module.get('enrollmentInfo', {}).get('count', 0),
                    "start_date": status_module.get('startDateStruct', {}).get('date', 'Unknown'),
                    "completion_date": status_module.get('completionDateStruct', {}).get('date', 'Unknown'),
                    "study_type": design_module.get('studyType', 'Unknown'),
                    "lead_sponsor": lead_sponsor,
                    "locations": locations
                }
                
                all_studies.append(study_obj)

            # Pausa di sicurezza per non martellare l'API governativa ed evitare blocchi IP
            time.sleep(1)

        except Exception as e:
            print(f"[ERRORE API] Fallimento per {drug_name}: {e}")

    # Salvataggio del file JSON piatto pronto per la successiva ingestione
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_studies, f, indent=4, ensure_ascii=False)

    print(f"\n[SUCCESSO] Download completato!")
    print(f"-> Totale Studi Clinici unici salvati: {len(all_studies)}")
    print(f"-> File salvato in: {output_file}")

if __name__ == "__main__":
    fetch_trials_to_json()