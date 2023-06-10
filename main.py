import spacy
from pathlib import Path
import typer
from spacy.tokens import DocBin

# Supported annotation labels
# TODO: Change it as required
INCL_ANN_LABELS = ['Dosage', 'Drug', 'Frequency']

def validateIfExactOneMatch(text_file_list, ann_filename):
    if len(text_file_list)>1:
        raise ValueError(f'Multiple text file matches for ann file {ann_filename}')
    elif len(text_file_list) == 0:
        raise ValueError(f'No match for ann file {ann_filename}')

def getTextFile(ann_filename, input_path):
    text_file_list = list(input_path.glob(f'{ann_filename.stem}.txt'))
    validateIfExactOneMatch(text_file_list, ann_filename)
    return text_file_list[0]

def getAnnToTextMap(input_path):
    ann_to_text_map = {}
    for ann_file in input_path.glob('*.ann'):
        ann_to_text_map[ann_file] = getTextFile(ann_file, input_path)
    return ann_to_text_map

def hasValidLength(ann):
    if len(ann) != 3:
        return False

def isTextAnnotation(ann):
    if not hasValidLength(ann):
        return False

    ann_type, _, _ = ann
    if ann_type[0] != 'T':
        return False
    return True

def getFilteredLabelIndex(label_idx):
    # No need to filter if semi colon not present in the label idx.
    if ';' not in label_idx:
        return label_idx.split()

    # If there's a word which has semi colon, remove that word.
    label_idx_filtered = []
    for word in label_idx.split():
        if ';' in word:
            continue
        label_idx_filtered.append(word)
    return label_idx_filtered

# Note: Text annotation is called entity
def getEntities(ann_file):
    entities = []
    ann_text = ann_file.read_text()
    for ann_line in ann_text.split('\n'):
        # Check if valid annotation
        ann = ann_line.split('\t')
        if not hasValidLength(ann) or not isTextAnnotation(ann):
            continue
        _, label_idx, text = ann

        # Filter label index
        filtered_label_idx = getFilteredLabelIndex(label_idx)
        ann_label, start_idx, end_idx = filtered_label_idx

        # Check if ann_label is the supported one.
        if ann_label in INCL_ANN_LABELS:
            entities.append([start_idx, start_idx, ann_label])
    return entities

def filterEntities(entities, doc):
    filtered_entities = []
    for entity in entities:
        entity_start_idx = int(entity[0])
        entity_end_idx = int(entity[1])
        entity_label = int(entity[2])
        span = doc.char_span(entity_start_idx, entity_end_idx, entity_label)
        if span:
            filtered_entities.append(span)
    # additionally some overlap, prefers longer spans
    filtered_entities = spacy.util.filter_spans(filtered_entities)
    return filtered_entities

def getFilteredEntities(ann_file, doc):
    entities = getEntities(ann_file)
    entities_filtered = filterEntities(entities, doc)
    return entities_filtered

def preprocess(input_path: Path):
    nlp = spacy.blank('en')
    doc_bin = DocBin(attrs=["ENT_IOB", "ENT_TYPE"])

    ann_to_text_map = getAnnToTextMap(input_path)
    for ann_file in ann_to_text_map:
        print('processing ann file: %s', ann_file)
        text_file = ann_to_text_map[ann_file]
        text = text_file.read_text()
        doc = nlp(text)
        entities = getFilteredEntities(ann_file, doc)
        doc.ents = entities
        doc_bin.add(doc)
    return doc_bin

def main(input_path: Path = typer.Argument(..., exists=True), output_path: Path = typer.Argument(...)):
    # Create doc bin from ann and text files.
    doc_bin = preprocess(input_path)

    # Store created doc bin locally.
    doc_bin.to_disk(output_path)

    print(f"Processed {len(doc_bin)} documents: {output_path.name}")

if __name__ == "__main__":
    typer.run(main)