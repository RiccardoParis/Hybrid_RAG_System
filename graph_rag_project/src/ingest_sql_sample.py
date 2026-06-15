import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

def populate_database():
    postgres_uri = os.getenv("POSTGRES_URI")
    engine = create_engine(postgres_uri)
    
    print("[Ingestion] Connessione al database...")
    
    try:
        with engine.connect() as conn:
            # 1. Pulizia tabelle esistenti
            conn.execute(text("DROP TABLE IF EXISTS used_cars;"))
            conn.execute(text("DROP TABLE IF EXISTS dealerships;"))
            
            # 2. Creazione schema
            print("[Ingestion] Creazione schema tabelle (dealerships, used_cars)...")
            conn.execute(text('''
                CREATE TABLE dealerships (
                    dealer_id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    state VARCHAR(50) NOT NULL,
                    rating FLOAT
                );
            '''))
            
            conn.execute(text('''
                CREATE TABLE used_cars (
                    car_id SERIAL PRIMARY KEY,
                    dealer_id INTEGER REFERENCES dealerships(dealer_id),
                    make VARCHAR(50) NOT NULL,
                    model VARCHAR(50) NOT NULL,
                    year INTEGER NOT NULL,
                    price DECIMAL(10, 2) NOT NULL,
                    mileage INTEGER NOT NULL,
                    condition VARCHAR(20)
                );
            '''))
            
            # 3. Inserimento dati mock
            print("[Ingestion] Inserimento dati di test...")
            conn.execute(text('''
                INSERT INTO dealerships (name, state, rating) VALUES 
                ('Craigslist Auto Direct', 'CA', 4.5),
                ('Texas Motors', 'TX', 4.2),
                ('NY Used Autos', 'NY', 3.8);
            '''))
            
            conn.execute(text('''
                INSERT INTO used_cars (dealer_id, make, model, year, price, mileage, condition) VALUES 
                (1, 'Toyota', 'Camry', 2018, 15500.00, 45000, 'Excellent'),
                (1, 'Honda', 'Civic', 2015, 11200.00, 85000, 'Good'),
                (2, 'Ford', 'F-150', 2020, 35000.00, 20000, 'Like New'),
                (2, 'Chevrolet', 'Silverado', 2017, 28000.00, 60000, 'Good'),
                (3, 'BMW', '3 Series', 2019, 24000.00, 30000, 'Excellent'),
                (3, 'Honda', 'Accord', 2014, 9500.00, 110000, 'Fair');
            '''))
            
            conn.commit()
            print("[Ingestion] Inserimento completato con successo!")
            
    except Exception as e:
        print(f"[Ingestion] Errore durante il popolamento: {e}")

if __name__ == "__main__":
    populate_database()
