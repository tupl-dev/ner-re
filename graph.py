import argparse
import json
import os
import sys
import torch
from neo4j import GraphDatabase, Query
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, GenerationConfig, pipeline, AutoTokenizer, Pipeline

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("--neo4j-uri", type=str, help="Neo4j URI", default=NEO4J_URI)
    parser.add_argument("--neo4j-user", type=str, help="Neo4j user", default=NEO4J_USER)
    parser.add_argument("--neo4j-password", type=str, help="Neo4j password", default=NEO4J_PASSWORD)
    return parser


def get_text(file: str) -> str:
    reader = PdfReader(file)
    return "".join([page.extract_text() for page in reader.pages])


def get_pipeline(cache: str = "./.cache/") -> Pipeline:
    cache_path = os.path.abspath(cache)
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda", cache_dir=cache_path)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def get_prompt(text: str) -> list:
    return [
        {
            "role": "system",
            "content": (
                "You are an information extraction bot. Analyze the text provided by the user. Identify key entities and find relations between them. "
                "Identify ONLY core biographical milestones (Birth, Parentage, Education, Key Employment, Major Accomplishments). Do not extract abstract concepts or minor details. "
                "Output the results ONLY as a structured JSON array of objects using keys: 'subject', 'subject_type', 'relation', 'object', 'object_type', 'original_text'. "
                "For 'original_text', copy the exact sentence or phrase from the source text that proves the extracted relationship. "
                "CRITICAL: If 'original_text' contains any double quotes (\"), you MUST escape them with a backslash (\\\"). "
            ),
        },
        {
            "role": "user",
            "content": f"Extract relations from this text:\n{text}",
        },
    ]


def parse_output(output: list) -> list:
    text = output[0]["generated_text"].strip()
    if text.startswith("```json"):
        text = text.removeprefix("```json").removesuffix("```").strip()
    elif text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").strip()

    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError as e:
        print(f"error loading json\n{e}", file=sys.stderr)
        print(f"output: {text}", file=sys.stderr)
        return []


def write_neo4j(data: list, uri: str, user: str, password: str):
    query = Query("""
        UNWIND $records AS record
        MERGE (s:Entity {name: record.subject})
        SET s.type = record.subject_type
        
        MERGE (o:Entity {name: record.object})
        SET o.type = record.object_type
        
        MERGE (s)-[r:RELATED_TO {relation: record.relation}]->(o)
        SET r.evidence = record.original_text
    """)

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        with driver.session() as session:
            session.run(query, records=data)


def main():
    args = get_args().parse_args()
    text = get_text(args.file)
    pipe = get_pipeline()
    prompt = get_prompt(text)

    print("Generating output...")
    config = GenerationConfig(max_new_tokens=4096, do_sample=False)
    output = pipe(prompt, return_full_text=False, generation_config=config)

    print("Generating graph...")
    parsed = parse_output(output)
    write_neo4j(parsed, args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    print(f"Generated graph, sent {len(parsed)} records")


if __name__ == "__main__":
    main()
