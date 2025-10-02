
import json
from rdflib import Graph, URIRef, RDFS, OWL
from rdflib.namespace import RDF
import os

def generate_ontology_qa_from_rdf(rdf_file_path, output_file_path):
    """Parses an RDF ontology file to extract concepts and their definitions,
    then generates a text-only Q&A dataset for knowledge infusion.
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
    
    # Define the types of concepts we are interested in
    concept_types = [URIRef(EDU_NS + 'Area'), URIRef(EDU_NS + 'Scope'), URIRef(EDU_NS + 'Ability')]
    
    qa_pairs = []

    # Find all individuals that are of the types we care about
    subjects = set()
    for concept_type in concept_types:
        for s, p, o in g.triples((None, RDF.type, concept_type)):
            # We only want to process the individuals, not the class definitions themselves
            if isinstance(s, URIRef) and s not in concept_types:
                 subjects.add(s)

    for s in sorted(list(subjects)):
        label_name = s.split('#')[-1]
        
        # Get definition from rdfs:isDefinedBy
        definition = g.value(subject=s, predicate=IS_DEFINED_BY)
        
        # Get all comments
        comments = list(g.objects(subject=s, predicate=COMMENT))
        
        output_parts = []
        
        # Prioritize isDefinedBy for the main definition
        if definition:
            output_parts.append(str(definition).strip())
        # If no isDefinedBy, use the first comment as definition (if available)
        elif comments:
            output_parts.append(str(comments.pop(0)).strip())

        # Append remaining comments as additional info/examples
        for comment in comments:
            # Prepend "Example:" if it's not already there, for clarity
            comment_text = str(comment).strip()
            if not comment_text.lower().startswith("example:"):
                output_parts.append(f"Example: {comment_text}")
            else:
                output_parts.append(comment_text)
            
        # Only create a pair if we found some descriptive text
        if output_parts:
            full_output = "\n\n".join(output_parts)
            
            qa_entry = {
                "instruction": f"What is the ontological concept '{label_name}'?",
                "output": full_output
            }
            qa_pairs.append(qa_entry)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        for entry in qa_pairs:
            f.write(json.dumps(entry) + '\n')
            
    print(f"Generated {len(qa_pairs)} Q&A pairs from the ontology in '{output_file_path}'.")

if __name__ == '__main__':
    rdf_path = os.path.join('ontology', 'core-ontology-0.4.0.rdf')
    output_path = 'ontology_qa.jsonl'
    generate_ontology_qa_from_rdf(rdf_path, output_path)