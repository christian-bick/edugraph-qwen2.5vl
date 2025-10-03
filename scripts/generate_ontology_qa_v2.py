
import json
from rdflib import Graph, URIRef, RDFS, OWL
from rdflib.namespace import RDF
import os

def generate_reversed_ontology_qa(rdf_file_path, output_file_path):
    """Parses an RDF ontology file to extract concepts and their definitions,
    then generates a text-only Q&A dataset where the instruction is the
    definition and the output is the structured JSON label.
    """
    if not os.path.exists(rdf_file_path):
        print(f"Error: RDF file not found at {rdf_file_path}")
        return

    g = Graph()
    g.parse(rdf_file_path)

    # Define namespaces and predicates
    EDU_NS = "http://edugraph.io/edu#"
    IS_DEFINED_BY = RDFS.isDefinedBy
    COMMENT = RDFS.comment
    
    concept_types = {
        "Areas": URIRef(EDU_NS + 'Area'),
        "Scopes": URIRef(EDU_NS + 'Scope'),
        "Abilities": URIRef(EDU_NS + 'Ability')
    }
    
    qa_pairs = []

    all_concepts = set()
    for type_uri in concept_types.values():
        for s, p, o in g.triples((None, RDF.type, type_uri)):
            if isinstance(s, URIRef) and s not in concept_types.values():
                 all_concepts.add(s)

    for s in sorted(list(all_concepts)):
        label_name = s.split('#')[-1]
        
        # Get definition from rdfs:isDefinedBy or rdfs:comment
        definition = g.value(subject=s, predicate=IS_DEFINED_BY)
        comments = list(g.objects(subject=s, predicate=COMMENT))
        
        instruction_text = ""
        if definition:
            instruction_text = str(definition).strip()
        elif comments:
            # Use the first comment as definition if isDefinedBy is missing
            instruction_text = str(comments.pop(0)).strip()
        
        # If there are remaining comments, append them as examples
        for comment in comments:
            instruction_text += f"\n\nExample: {str(comment).strip()}"

        # Skip if we couldn't find a definition
        if not instruction_text:
            continue

        # Determine the type and construct the JSON output
        json_output = {"Areas": [], "Scopes": [], "Abilities": []}
        concept_type_found = False
        for type_name, type_uri in concept_types.items():
            if (s, RDF.type, type_uri) in g:
                json_output[type_name].append(label_name)
                concept_type_found = True
                break # Assume each concept has only one primary type for this structure
        
        if concept_type_found:
            qa_entry = {
                "instruction": instruction_text,
                "output": json.dumps(json_output)
            }
            qa_pairs.append(qa_entry)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in qa_pairs:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Generated {len(qa_pairs)} reversed Q&A pairs in '{output_file_path}'.")

if __name__ == '__main__':
    rdf_path = os.path.join('ontology', 'core-ontology-0.4.0.rdf')
    output_path = 'ontology_qa_v2.jsonl'
    generate_reversed_ontology_qa(rdf_path, output_path)
