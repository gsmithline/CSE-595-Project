import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import numpy as np
import xml.etree.ElementTree as ET
import utils
from fuzzywuzzy import fuzz
'''
def extract_title(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

    # Extract the title
    title_elem = root.find('.//tei:titleStmt/tei:title[@level="a"][@type="main"]', namespaces)
    title = title_elem.text.lower() if title_elem is not None and title_elem.text else "title not found"
    return title
'''
def extract_title(file_path):
    # Parse the XML file
    try:
        tree = ET.parse(file_path)
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return "title not found"

    root = tree.getroot()
    namespaces = {'tei': 'http://www.tei-c.org/ns/1.0'}

    # Extract the title
    title_elem = root.find('.//tei:titleStmt/tei:title[@level="a"][@type="main"]', namespaces)
    if title_elem is not None and title_elem.text:
        title = title_elem.text.lower()
    else:
        title = "title not found"
    return title

def build_title2id_dict():
    # Build a dictionary mapping paper titles to their IDs
    data_dir = '../data'
    papers = utils.load_json(data_dir, "paper_source_trace_train.json")  # Adjust the filename if needed
    title2id_dict = {}
    for paper in papers:
        pid = paper["_id"]
        title = paper.get("title", "").lower()
        title2id_dict[title] = pid
    return title2id_dict
def main():
    title2id_dict = build_title2id_dict()
    data_dir = '../data'
    papers = utils.load_json(data_dir, "paper_source_trace_valid_wo_ans.json")
    sub_example_dict = utils.load_json(data_dir, "submission_example_valid.json")
    embed_dir = os.path.join(data_dir, "paper_embed_openai")
    xml_dir = os.path.join(data_dir, "paper-xml")
    sub_dict = {}

    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        cur_embed = np.load(os.path.join(embed_dir, cur_pid + '.npy'))
        file = os.path.join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        n_refs = 0
        print(f"processing {cur_pid}")
        for ref in references:
            print(f"processing ref {ref}")
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                print(f"ref {ref} has no analytic")
                continue
            if ref.analytic.title is None:
                print(f"ref {ref} has no title")
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                print(f"ref {ref} has index {b_idx} which is greater than n_refs {n_refs}")
                n_refs = b_idx
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]
        y_score = [0] * n_refs
        assert len(sub_example_dict[cur_pid]) == n_refs

        predicted_scores = []
        print(f"processing {cur_pid} with {n_refs} refs")
        for test_id in bib_sorted:
            print(f"processing {test_id}")
            if test_id not in bid_to_title.keys():
                print(f"ref {test_id} not in bid_to_title")
                predicted_scores.append(0)
                continue
            test_title = bid_to_title[test_id]
            test_pid = None
            for title in title2id_dict.keys():
                if fuzz.ratio(title, test_title) >= 80:
                    
                    test_pid = title2id_dict[title]
                    break
            if test_pid is None:
                print(f"ref {test_id} not in title2id_dict")
                predicted_scores.append(0)
                continue
            test_embed = np.load(os.path.join(embed_dir, test_pid + '.npy'))
            score = np.dot(cur_embed, test_embed) / np.linalg.norm(cur_embed) / np.linalg.norm(test_embed)
            predicted_scores.append(score)
            print(f"score for {test_id} is {score}")
            
        for ii in range(len(predicted_scores)):
            print(f"processing {bib_sorted[ii]}")
            bib_idx = int(bib_sorted[ii][1:])
            y_score[bib_idx] = predicted_scores[ii]
            print(f"y_score for {bib_sorted[ii]} is {y_score[bib_idx]}")
        sub_dict[cur_pid] = y_score
        print(f"sub_dict for {cur_pid} is {sub_dict[cur_pid]}")
    
    utils.dump_json(sub_dict, '../data', "valid_submission_scibert.json")


if __name__ == "__main__":
    main()