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
warnings.filterwarnings('ignore')

from joblib import Parallel, delayed, cpu_count
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
    backbone = backbone[(backbone.chain_id == chain_id)]
    
    res_id = np.unique(backbone.res_id)
    res_id2idx = {id:idx for idx, id in enumerate(res_id)}
    res_num = res_id.shape[0]
    coords = np.zeros((res_num,len(atom_names),3)) + np.nan
    for idx, name in enumerate(atom_names):
        mask = backbone.atom_name == name
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
    

def extract_data_from_ec(protein_name, label):
    data = {
        "seq": "",
        "coords": {
            "N": [],
            "CA": [],
            "C": [],
            "O": []
        },
        "name": protein_name,
        "label":label
    }
    protein_name, chain_name = protein_name.split("-")

    pdb_path = '/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/pdb'
    cif_path = '/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/cif'
    cif_url = f"https://files.rcsb.org/download/{protein_name}.cif"
    pdb_url = f"https://files.rcsb.org/download/{protein_name}.pdb"

    cif_file_path = os.path.join(cif_path, f"{protein_name}.cif")
    pdb_file_path = os.path.join(pdb_path, f"{protein_name}.pdb")
    
    if not os.path.exists(cif_file_path):
        try:
            if download_file(cif_url, cif_file_path):
                print(f"Successfully downloaded {protein_name}.cif")
                # os.remove(cif_file_path)  
            else:
                raise Exception("Failed to download CIF file")
        except Exception as e:
            print(f"Error: {e}, switching to pdb")
            if not os.path.exists(pdb_file_path):
                if download_file(pdb_url, pdb_file_path):
                    print(f"Successfully downloaded {protein_name}.pdb")
                    # os.remove(pdb_file_path)  
                else:
                    print(f"Failed to download both CIF and PDB files for {protein_name}")
    
    seq, coords = get_seq_coords(cif_file_path, chain_name, ['N','CA','C','O'])
    
    data['seq'] = seq
    data['coords']['N'] = np.around(coords[:, 0], decimals=3).tolist()
    data['coords']['CA'] = np.around(coords[:, 1], decimals=3).tolist()
    data['coords']['C'] = np.around(coords[:, 2], decimals=3).tolist()
    data['coords']['O'] = np.around(coords[:, 3], decimals=3).tolist()

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', type=str)
    args = parser.parse_args()
    
    tasks = []
    ec_annotations = {}
    ec_num = {}
    labels = {}
    level_idx = 1
    dict_list = []
    # Initialize counter
    ec_cnt = 0

    # Open the file
    with open('/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/ec/nrPDB-EC_annot.tsv', 'r') as f:
        for idx, line in enumerate(f):
            # Handle the second line of the file
            if idx == 1:
                arr = line.rstrip().split('\t')
                for ec in arr:
                    ec_annotations[ec] = ec_cnt
                    ec_num[ec] = 0
                    ec_cnt += 1

            # Handle the fourth line and beyond
            elif idx > 2:
                arr = line.rstrip().split('\t')
                protein_labels = []
                if len(arr) > level_idx:
                    protein_ec_list = arr[level_idx].split(',')
                    for ec in protein_ec_list:
                        if len(ec) > 0:
                            protein_labels.append(ec_annotations[ec])
                            ec_num[ec] += 1
                labels[arr[0]] = protein_labels
    protein_seq_map = {}
    corrs_label = []
    reference_seqs = {}
    with open(f'/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/ec/{args.split}.fasta', 'r') as f:
        for line in f:
            if line.startswith('>'):
                protein_name = line.rstrip()[1:]
                tasks.append(protein_name)
                corrs_label.append(labels.get(protein_name,""))
            else:
                reference_seqs[protein_name] = line.rstrip()
                

    tasks_labels = list(zip(tasks,corrs_label))
    
   
    batch = pmap_multi(extract_data_from_ec, tasks_labels, n_jobs=-1)

    with open(f"/gaozhangyang/zyj/OpenCPD_MUT/data/ec/ec_{args.split}.jsonl", "w") as file:
        for dictionary in batch:
            json.dump(dictionary, file)
            file.write('\n')
    

