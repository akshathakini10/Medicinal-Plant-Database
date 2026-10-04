import json
import re
from pathlib import Path
from neo4j import GraphDatabase

# Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = PROJECT_ROOT / "data" / "integrated" / "medicinal_knowledge_base.json"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password123"

def extract_diseases(shloka_list):
    diseases = set()
    indicators = [
        "kustha", "krimi", "krmi", "sopha", "jvara", "arsa", "prameha",
        "kasa", "svasa", "skin disease", "fever", "cough", "asthma",
        "inflammation", "ulcer", "diabetes", "diarrhea", "piles"
    ]
    for item in shloka_list:
        text = ""
        if isinstance(item, dict):
            text = " ".join([str(v) for v in item.values()]).lower()
        elif isinstance(item, str):
            text = item.lower()
        for ind in indicators:
            if ind in text:
                diseases.add(ind.strip())
    return list(diseases)

def run_ingestion():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Plant) REQUIRE p.id UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Family) REQUIRE f.name UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disease) REQUIRE d.name UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (prop:AyurvedicProperty) REQUIRE prop.name UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (h:Habitat) REQUIRE h.name UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:System) REQUIRE s.name UNIQUE"
    ]

    with driver.session() as session:
        for c in constraints:
            session.run(c)
        print("[+] Schema constraints initialized.")

    cypher_query = """
    UNWIND $batch AS row
    MERGE (p:Plant {id: row.plant_id})
    SET p.botanical_name = row.botanical_name,
        p.vernacular_names = row.vernacular_names

    FOREACH (fam IN CASE WHEN row.family <> '' AND row.family <> 'Not available' THEN [row.family] ELSE [] END |
        MERGE (f:Family {name: toLower(fam)})
        MERGE (p)-[:BELONGS_TO_FAMILY]->(f)
    )
    FOREACH (sys IN row.systems |
        MERGE (s:System {name: toLower(trim(sys))})
        MERGE (p)-[:USED_IN_SYSTEM]->(s)
    )
    FOREACH (prop IN row.ayurvedic_properties |
        MERGE (ap:AyurvedicProperty {name: toLower(trim(prop.name))})
        SET ap.type = prop.type
        MERGE (p)-[:HAS_PROPERTY]->(ap)
    )
    FOREACH (d IN row.extracted_diseases |
        MERGE (dis:Disease {name: toLower(trim(d))})
        MERGE (p)-[:TREATS]->(dis)
    )
    FOREACH (state IN row.states |
        MERGE (h:Habitat {name: toLower(trim(state))})
        MERGE (p)-[:FOUND_IN]->(h)
    )
    """

    batch = []
    batch_size = 100
    with driver.session() as session:
        for plant in data:
            ayur_props = []
            for item in plant.get("ayurvedic_identity", []):
                if isinstance(item, dict):
                    for prop_type in ["rasa", "virya", "vipaka"]:
                        if item.get(prop_type):
                            ayur_props.append({"name": str(item[prop_type]), "type": prop_type.capitalize()})

            dist = plant.get("distribution", {})
            states = []
            if isinstance(dist, dict):
                raw_states = dist.get("states", "") or dist.get("state", "")
                if isinstance(raw_states, str):
                    states = [s.strip() for s in re.split(r"[,;]", raw_states) if s.strip()]

            batch.append({
                "plant_id": str(plant.get("plant_id", "")),
                "botanical_name": plant.get("botanical_name", "Unknown"),
                "family": plant.get("family", ""),
                "vernacular_names": str(plant.get("vernacular_names", "")),
                "systems": plant.get("systems", []),
                "ayurvedic_properties": ayur_props,
                "extracted_diseases": extract_diseases(plant.get("shlokas", [])),
                "states": states
            })

            if len(batch) >= batch_size:
                session.run(cypher_query, batch=batch)
                batch = []

        if batch:
            session.run(cypher_query, batch=batch)

    driver.close()
    print("[+] Neo4j Knowledge Graph successfully populated.")

if __name__ == "__main__":
    run_ingestion()