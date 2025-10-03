
import json
from rdflib import Graph, URIRef, RDFS, OWL, Namespace
from rdflib.namespace import RDF
import os
from collections import defaultdict

def generate_full_qa(rdf_file_path, output_file_path):
    """Parses an RDF ontology file to generate a comprehensive Q&A dataset
    including definition-to-label and hierarchy questions.
    """
    if not os.path.exists(rdf_file_path):
        print(f"Error: RDF file not found at {rdf_file_path}")
        return

    g = Graph()
    g.parse(rdf_file_path)

    EDU = Namespace("http://edugraph.io/edu#")
    IS_DEFINED_BY = RDFS.isDefinedBy
    COMMENT = RDFS.comment

    concept_types = {
        "Areas": EDU.Area,
        "Scopes": EDU.Scope,
        "Abilities": EDU.Ability
    }

    qa_pairs = []

    # --- Part 1: Generate Definition -> JSON pairs ---
    all_concepts = set()
    for type_uri in concept_types.values():
        for s, _, _ in g.triples((None, RDF.type, type_uri)):
            if isinstance(s, URIRef) and s not in concept_types.values():
                all_concepts.add(s)

    for s in sorted(list(all_concepts)):
        label_name = s.split('#')[-1]
        definition = g.value(subject=s, predicate=IS_DEFINED_BY)
        comments = list(g.objects(subject=s, predicate=COMMENT))
        
        instruction_text = ""
        if definition:
            instruction_text = str(definition).strip()
        elif comments:
            instruction_text = str(comments.pop(0)).strip()
        
        if not instruction_text:
            continue

        json_output = {"Areas": [], "Scopes": [], "Abilities": []}
        type_found = False
        for type_name, type_uri in concept_types.items():
            if (s, RDF.type, type_uri) in g:
                json_output[type_name].append(label_name)
                type_found = True
                break
        
        if type_found:
            qa_pairs.append({
                "instruction": instruction_text,
                "output": json.dumps(json_output)
            })

    # --- Part 2: Generate Hierarchy pairs ---
    parent_to_children = defaultdict(list)
    child_to_parent = {}
    part_of_predicates = [EDU.partOfArea, EDU.partOfScope, EDU.partOfAbility]

    for pred in part_of_predicates:
        for child, _, parent in g.triples((None, pred, None)):
            parent_to_children[parent].append(child)
            child_to_parent[child] = parent

    # Child -> Parent questions
    for child, parent in child_to_parent.items():
        child_name = child.split('#')[-1]
        parent_name = parent.split('#')[-1]
        
        json_output = {"Areas": [], "Scopes": [], "Abilities": []}
        type_found = False
        for type_name, type_uri in concept_types.items():
            if (parent, RDF.type, type_uri) in g:
                json_output[type_name].append(parent_name)
                type_found = True
                break
        
        if type_found:
            qa_pairs.append({
                "instruction": f"What is the parent of the concept '{child_name}'?",
                "output": json.dumps(json_output)
            })

    # Parent -> Children questions
    for parent, children in parent_to_children.items():
        parent_name = parent.split('#')[-1]
        child_names = sorted([c.split('#')[-1] for c in children])

        json_output = {"Areas": [], "Scopes": [], "Abilities": []}
        type_found = False
        for type_name, type_uri in concept_types.items():
            if (parent, RDF.type, type_uri) in g:
                json_output[type_name] = child_names
                type_found = True
                break

        if type_found:
            qa_pairs.append({
                "instruction": f"What are the children of the concept '{parent_name}'?",
                "output": json.dumps(json_output)
            })

    # --- Write to file ---
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in qa_pairs:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Generated {len(qa_pairs)} comprehensive Q&A pairs in '{output_file_path}'.")

if __name__ == '__main__':
    rdf_path = os.path.join('ontology', 'core-ontology-0.4.0.rdf')
    output_path = 'ontology_qa_v3.jsonl'
    generate_full_qa(rdf_path, output_path)
