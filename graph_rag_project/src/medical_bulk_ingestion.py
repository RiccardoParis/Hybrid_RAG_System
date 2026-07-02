import os
import glob
import json
import sys
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

# Importiamo le funzioni che hai già per Neo4j e Qdrant
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ingest import ingest_document_to_vector, ingest_graph_from_json

load_dotenv()
POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://postgres:Password@127.0.0.1:5433/medical_rag_db")

Base = declarative_base()

# --- DEFINIZIONE SCHEMA SQL ---
study_sponsor_table = Table(
    'study_sponsors', Base.metadata,
    Column('study_nct_id', String, ForeignKey('studies.nct_id'), primary_key=True),
    Column('sponsor_id', Integer, ForeignKey('sponsors.id'), primary_key=True)
)

class Drug(Base):
    __tablename__ = 'drugs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    studies = relationship('Study', back_populates='drug')

class Study(Base):
    __tablename__ = 'studies'
    nct_id = Column(String, primary_key=True)
    drug_id = Column(Integer, ForeignKey('drugs.id'))
    title = Column(String)
    status = Column(String)
    phase = Column(String)
    enrollment = Column(Integer)
    start_date = Column(String)
    completion_date = Column(String)
    study_type = Column(String)
    
    drug = relationship('Drug', back_populates='studies')
    sponsors = relationship('Sponsor', secondary=study_sponsor_table, back_populates='studies')
    locations = relationship('Location', back_populates='study')

class Sponsor(Base):
    __tablename__ = 'sponsors'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False)
    studies = relationship('Study', secondary=study_sponsor_table, back_populates='sponsors')

class Location(Base):
    __tablename__ = 'locations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    study_id = Column(String, ForeignKey('studies.nct_id'))
    facility = Column(String)
    city = Column(String)
    country = Column(String)
    study = relationship('Study', back_populates='locations')

# --- FUNZIONE INGESTIONE SQL ---
def ingest_sql_from_json(json_path):
    print(f"[SQL] Connessione a PostgreSQL e reset tabelle...")
    engine = create_engine(POSTGRES_URI)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    
    Session = sessionmaker(bind=engine)
    session = Session()

    print(f"[SQL] Caricamento dati da {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sponsor_cache = {}
    drug_cache = {}

    for row in data:
        # 1. Farmaco
        d_name = row.get('drug_name', 'Unknown')
        if d_name not in drug_cache:
            drug = Drug(name=d_name)
            session.add(drug)
            session.flush()
            drug_cache[d_name] = drug
            
        drug_obj = drug_cache[d_name]

        # Evita duplicati di trial
        if session.query(Study).filter_by(nct_id=row['nct_id']).first():
            continue

        # 2. Studio Clinico
        study = Study(
            nct_id=row['nct_id'],
            drug_id=drug_obj.id,
            title=row.get('title', 'Unknown')[:250], # Troncamento sicurezza
            status=row.get('status', 'Unknown'),
            phase=row.get('phase', 'Unknown'),
            enrollment=row.get('enrollment', 0),
            start_date=row.get('start_date', 'Unknown'),
            completion_date=row.get('completion_date', 'Unknown'),
            study_type=row.get('study_type', 'Unknown')
        )
        session.add(study)

        # 3. Sponsor
        sp_name = row.get('lead_sponsor', 'Unknown')[:100]
        if sp_name not in sponsor_cache:
            sponsor = Sponsor(name=sp_name)
            session.add(sponsor)
            session.flush()
            sponsor_cache[sp_name] = sponsor
            
        study.sponsors.append(sponsor_cache[sp_name])

        # 4. Locations
        for loc in row.get('locations', []):
            location = Location(
                study_id=study.nct_id,
                facility=loc.get('facility', 'Unknown')[:150],
                city=loc.get('city', 'Unknown')[:100],
                country=loc.get('country', 'Unknown')[:100]
            )
            session.add(location)

    session.commit()
    print(f"[SQL] Ingestione PostgreSQL completata! Salvati {len(data)} trial clinici normalizzati.")

# --- ORCHESTRATORE PRINCIPALE ---
def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    texts_dir = os.path.join(base_dir, "data", "texts")
    tables_dir = os.path.join(base_dir, "data", "tables")
    graphs_dir = os.path.join(base_dir, "data", "graphs")

    print("=== AVVIO INGESTIONE MASSIVA DOMINIO MEDICO ===")

    # 1. SQL Ingestion
    print("\n--- 1. INGESTIONE POSTGRESQL (Trial Clinici) ---")
    sql_json = os.path.join(tables_dir, "clinical_trials_data.json")
    if os.path.exists(sql_json):
        ingest_sql_from_json(sql_json)
    else:
        print(f"[ATTENZIONE] File {sql_json} non trovato.")

    # 2. Graph Ingestion
    print("\n--- 2. INGESTIONE NEO4J (Knowledge Graph) ---")
    graph_json = os.path.join(graphs_dir, "neurology_graph.json")
    if os.path.exists(graph_json):
        ingest_graph_from_json(graph_json)
    else:
        print(f"[ATTENZIONE] File {graph_json} non trovato.")

    # 3. Vector Ingestion
    print("\n--- 3. INGESTIONE QDRANT (Paper PubMed) ---")
    text_files = glob.glob(os.path.join(texts_dir, "*.txt"))
    if not text_files:
        print(f"[ATTENZIONE] Nessun file di testo trovato in {texts_dir}.")
    else:
        for idx, file_path in enumerate(text_files, 1):
            filename = os.path.basename(file_path)
            print(f"[{idx}/{len(text_files)}] Vettorizzazione: {filename}...")
            ingest_document_to_vector(file_path)
        print("[VECTOR] Ingestione Qdrant completata!")

    print("\n=== TUTTI I DATI SONO STATI INGERITI CON SUCCESSO! ===")

if __name__ == "__main__":
    main()