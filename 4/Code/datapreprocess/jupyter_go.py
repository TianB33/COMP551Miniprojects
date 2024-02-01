# import h5py
import numpy as np
import json
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import three_to_one
import os
import json
import numpy as np
import re
import requests
from Bio.PDB import PDBList
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import requests
from biotite.structure.io.pdbx import PDBxFile, get_structure
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
import argparse
import warnings


from joblib import Parallel, delayed, cpu_count

warnings.filterwarnings('ignore')

AA_NAME_SYM = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
            'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
            'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'UNK': "X",'SEC':"X"
        }
        
get_restype = lambda res_name: AA_NAME_SYM[res_name] if res_name in AA_NAME_SYM else 'X'

def get_seq_coords(file_path, chain_id='A', atom_names=['CA']):
    
    if file_path.endswith(".cif"):
        source = PDBxFile.read(file_path)
        chain = get_structure(source, model=1)
    elif file_path.endswith('.pdb'):
        structure = PDBFile.read(file_path)
        chain = structure.get_structure()[0]
    
    backbone = chain[struc.filter_backbone(chain)]
    
    
    res_id = np.unique(backbone.res_id)
    res_id2idx = {id:idx for idx, id in enumerate(res_id)}
    res_num = res_id.shape[0]
    coords = np.zeros((res_num,len(atom_names),3)) + np.nan
    for idx, name in enumerate(atom_names):
        mask = (backbone.chain_id == chain_id)&(backbone.atom_name == name)
        atoms = backbone[mask]
        
        index = [res_id2idx[one] for one in atoms.res_id]
        coords[index,idx] = atoms.coord

    
    seqs = np.array(['UNK' for idx in range(res_num)])
    _, uni_idx = np.unique(backbone.res_id, return_index=True)
    index = [res_id2idx[one] for one in backbone.res_id[uni_idx]]
    seqs[index] = backbone[uni_idx].res_name
    
    seq = "".join([get_restype(res_name) for res_name in seqs])
    
    assert len(seq) == len(coords), "the length of seq do not match that of coords!"
    return seq, coords

def pmap_multi(pickleable_fn, data, n_jobs=None, verbose=1, desc=None, **kwargs):
    """

    Parallel map using joblib.

    Parameters
    ----------
    pickleable_fn : callable
        Function to map over data.
    data : iterable
        Data over which we want to parallelize the function call.
    n_jobs : int, optional
        The maximum number of concurrently running jobs. By default, it is one less than
        the number of CPUs.
    verbose: int, optional
        The verbosity level. If nonzero, the function prints the progress messages.
        The frequency of the messages increases with the verbosity level. If above 10,
        it reports all iterations. If above 50, it sends the output to stdout.
    kwargs
        Additional arguments for :attr:`pickleable_fn`.

    Returns
    -------
    list
        The i-th element of the list corresponds to the output of applying
        :attr:`pickleable_fn` to :attr:`data[i]`.
    """
    if n_jobs is None:
        n_jobs = cpu_count() - 1
    # n_jobs = 60
    results = Parallel(n_jobs=n_jobs, verbose=verbose, timeout=None)(
    delayed(pickleable_fn)(*d, **kwargs) for i, d in tqdm(enumerate(data),desc=desc)
    )

    return results

def download_file(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    else:
        return False
    
def get_coords_or_nan(residue, atom_name):
    return list(map(convert_float32, residue[atom_name].get_coord())) if atom_name in residue else [np.nan, np.nan, np.nan]   



def convert_float32(value):
    if isinstance(value, np.float32):
        return round(float(value), 3)
    else:
        return value

def extract_data_from_ec(protein_name, label_map):
    data = {
        "seq": "",
        "coords": {
            "N": [],
            "CA": [],
            "C": [],
            "O": []
        },
        "name": protein_name,
        "chain_encoding": {},
        "label": label_map
    }
    
    protein_name, chain_name = protein_name.split("-")
    if protein_name == "4GFV":
        print('ddddddddddddddddddddd')
    pdb_path = '/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/pdb'
    cif_path = '/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/cif'
    cif_url = f"https://files.rcsb.org/download/{protein_name}.cif"
    pdb_url = f"https://files.rcsb.org/download/{protein_name}.pdb"

    cif_file_path = os.path.join(cif_path, f"{protein_name}.cif")
    pdb_file_path = os.path.join(pdb_path, f"{protein_name}.pdb")
    
    if os.path.exists(cif_file_path):
        # print(f"{protein_name}.cif already exists, skipping download.")
        parser = MMCIFParser()
        structure = parser.get_structure("pdb_structure", cif_file_path)
    else:
        try:
            if download_file(cif_url, cif_file_path):
                print(f"Successfully downloaded {protein_name}.cif")
                parser = MMCIFParser()
                structure = parser.get_structure("pdb_structure", cif_file_path)
                # os.remove(cif_file_path)  
            else:
                raise Exception("Failed to download CIF file")
        except Exception as e:
            print(f"Error: {e}, switching to pdb")
            if os.path.exists(pdb_file_path):
                print(f"{protein_name}.pdb already exists, skipping download.")
                parser = PDBParser()
                structure = parser.get_structure("pdb_structure", pdb_file_path)
            elif download_file(pdb_url, pdb_file_path):
                print(f"Successfully downloaded {protein_name}.pdb")
                parser = PDBParser()
                structure = parser.get_structure("pdb_structure", pdb_file_path)
                # os.remove(pdb_file_path) 
            else:
                print(f"Failed to download both CIF and PDB files for {protein_name}")
        
    for model in structure:
        chain_id = []
        idx = 0
        for chain in model:
            if chain.id != chain_name:
                continue
            idx += 1
            chain_id.append(idx)
            res_num = []
            res_idx = 0
            
            for residue in chain:
                #print(pdb_file)
                #print(residue)
                if residue.id[0] != " " or residue.get_id()[0] != " ": 
                    continue  
                if all(atom in residue for atom in ["CA"]):
                    res_idx += 1

                    three_letter_code = residue.get_resname()
                    one_letter_code = 'X' if three_letter_code in ["UNK", "SEC"] else three_to_one(three_letter_code)
                    data["seq"] += one_letter_code

        
                    for atom_name in ["N", "CA", "C", "O"]:
                        coords = get_coords_or_nan(residue, atom_name)
                        data["coords"][atom_name].append(coords)

                res_num.append(res_idx)
            if len(data["seq"]) != len(data['coords']['CA']):
                print('---------------------------------------------------------------------------')
            data["chain_encoding"][idx] = res_idx
        break
    return data

# Load the fasta file.
# Initialize dictionaries
ec_annotations = {}
ec_num = {}
labels = {}
level_idx = 1
dict_list = []
# Initialize counter
ec_cnt = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', type=str)
    args = parser.parse_args()
    
    tasks = []
    
    # Load the fasta file.
# Initialize dictionaries
    ec_annotations = {}
    ec_num = {}
    labels = {}
    level_idx = 1
    dict_list = []
    # Initialize counter
    ec_cnt = 0
    level = 'cc'
    level_idx = 0
    go_cnt = 0
    go_num = {}
    go_annotations = {}
    labels = {}
    # Open the file
    with open('/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/go/nrPDB-GO_2019.06.18_annot.tsv', 'r') as f:
        for idx, line in enumerate(f):
            if idx == 1 and level == "mf":
                level_idx = 1
                arr = line.rstrip().split('\t')
                for go in arr:
                    go_annotations[go] = go_cnt
                    go_num[go] = 0
                    go_cnt += 1
            elif idx == 5 and level == "bp":
                level_idx = 2
                arr = line.rstrip().split('\t')
                for go in arr:
                    go_annotations[go] = go_cnt
                    go_num[go] = 0
                    go_cnt += 1
            elif idx == 9 and level == "cc":
                level_idx = 3
                arr = line.rstrip().split('\t')
                for go in arr:
                    go_annotations[go] = go_cnt
                    go_num[go] = 0
                    go_cnt += 1
            elif idx > 12:
                arr = line.rstrip().split('\t')
                protein_labels = []
                if len(arr) > level_idx:
                    protein_go_list = arr[level_idx]
                    protein_go_list = protein_go_list.split(',')
                    for go in protein_go_list:
                        if len(go) > 0:
                            protein_labels.append(go_annotations[go])
                            go_num[go] += 1
                labels[arr[0]] = protein_labels

    corrs_label = []
    with open(f'/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/go/{args.split}.fasta', 'r') as f:
        for line in f:
            if line.startswith('>'):
                protein_name = line.rstrip()[1:]
                tasks.append(protein_name)
                corrs_label.append(labels.get(protein_name,""))
                

    tasks_labels = list(zip(tasks,corrs_label))
    tasks_labels = [one for one in tasks_labels if one[0]=='5EXC-I']

    batch = pmap_multi(extract_data_from_ec, tasks_labels, n_jobs=1)

    with open(f"/gaozhangyang/zyj/OpenCPD_MUT/data/go/{level}_{args.split}.jsonl", "w") as file:
        for dictionary in batch:
            json.dump(dictionary, file)
            file.write('\n')
