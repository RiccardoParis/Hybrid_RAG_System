import os
import requests
import time
import xml.etree.ElementTree as ET

def fetch_pubmed_abstracts():
    # Creiamo la cartella di output per i file di testo
    output_dir = "../data/texts"
    os.makedirs(output_dir, exist_ok=True)

    target_drugs = [
        "Apomorphine", "Benzatropine", "Trihexyphenidyl", "Atropine", "Ropinirole", 
        "Gabapentin", "Fludrocortisone", "Rasagiline", "Memantine", "Bromocriptine", 
        "Entacapone", "Clonazepam", "Rivastigmine", "Galantamine", "Diphenhydramine", 
        "Carbidopa", "Pergolide", "Amantadine", "Cabergoline", "Modafinil", "L-DOPA", 
        "Rotigotine", "Procyclidine", "Donepezil", "Ramelteon", "Pramipexole", 
        "Biperiden", "Haloperidol", "Tolcapone", "Hyoscyamine", "Quetiapine", "Selegiline"
    ]

    # Limite di paper per farmaco (50 è un ottimo numero per avere volume senza bloccare il PC locale)
    MAX_PAPERS_PER_DRUG = 50
    total_saved = 0

    print(f"Inizio download paper accademici da PubMed per {len(target_drugs)} farmaci...")

    for drug in target_drugs:
        print(f"\n[PubMed] Cerco i {MAX_PAPERS_PER_DRUG} paper più rilevanti per: {drug}...")
        
        # 1. Ricerca degli ID dei paper (PMID) tramite esearch
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": f"{drug}[Title/Abstract] AND (Parkinson OR Alzheimer)",
            "retmode": "json",
            "retmax": MAX_PAPERS_PER_DRUG,
            "sort": "relevance"
        }

        try:
            search_response = requests.get(search_url, params=search_params)
            search_response.raise_for_status()
            pmids = search_response.json().get("esearchresult", {}).get("idlist", [])
            
            if not pmids:
                print(f" -> Nessun paper trovato per {drug}.")
                continue
                
            print(f" -> Trovati {len(pmids)} PMIDs. Inizio estrazione degli abstract...")

            # 2. Download degli Abstract tramite efetch
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml"
            }
            
            fetch_response = requests.get(fetch_url, params=fetch_params)
            fetch_response.raise_for_status()
            
            # Parsing dell'XML restituito da PubMed
            root = ET.fromstring(fetch_response.content)
            
            saved_for_drug = 0
            for article in root.findall(".//PubmedArticle"):
                # Estrazione PMID
                pmid = article.find(".//PMID").text if article.find(".//PMID") is not None else "unknown"
                
                # Estrazione Titolo
                title_node = article.find(".//ArticleTitle")
                title = title_node.text if title_node is not None else "No Title"
                
                # Estrazione Abstract
                abstract_texts = article.findall(".//AbstractText")
                if abstract_texts:
                    abstract = " ".join([node.text for node in abstract_texts if node.text])
                    
                    if abstract.strip():
                        # Salviamo il contenuto in un file .txt
                        filename = os.path.join(output_dir, f"{drug.replace(' ', '_')}_{pmid}.txt")
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"Title: {title}\n")
                            f.write(f"Drug: {drug}\n")
                            f.write(f"Source: PubMed (PMID: {pmid})\n\n")
                            f.write(f"Abstract:\n{abstract}\n")
                        
                        saved_for_drug += 1
                        total_saved += 1
            
            print(f" -> Salvati {saved_for_drug} documenti testuali per {drug}.")
            time.sleep(2) # Rispetto dei limiti API di NCBI (massimo 3 richieste al secondo)

        except Exception as e:
            print(f"[ERRORE] Fallimento per {drug}: {e}")

    print(f"\n[SUCCESSO] Download completato!")
    print(f"-> Totale documenti testuali generati in {output_dir}: {total_saved}")

if __name__ == "__main__":
    fetch_pubmed_abstracts()