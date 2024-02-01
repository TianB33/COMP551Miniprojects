import h5py
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
from collections import defaultdict
warnings.filterwarnings('ignore')
from Bio import pairwise2
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
from Bio import pairwise2

def recover_seq(aligned_ori_seq, aligned_seq):
    aligned_ori_seq = list(aligned_ori_seq)
    aligned_seq = list(aligned_seq)
    new_seq = []
    assert len(aligned_ori_seq) == len(aligned_seq)
    
    for i in range(len(aligned_ori_seq)):
        if i < len(aligned_ori_seq)-1:
            
            condition1 = aligned_ori_seq[i] == '-' and aligned_seq[i] != '-' and aligned_seq[i-1] == '-' and aligned_ori_seq[i-1] != '-'\
                and aligned_ori_seq[i+1] == aligned_seq[i+1]
            condition2 = aligned_ori_seq[i] == '-' and aligned_seq[i] != '-' and aligned_seq[i-1] == aligned_ori_seq[i-1]\
                and aligned_ori_seq[i+1] == aligned_seq[i+1]
            condition3 = aligned_ori_seq[i] != '-' and aligned_seq[i] == '-' and aligned_seq[i-1] == aligned_ori_seq[i-1]\
                and aligned_ori_seq[i+1] == aligned_seq[i+1]
            
            if condition1:
                pass  # Skip this iteration, effectively removing the element in new_seq
            elif condition2:
                new_seq.append(aligned_seq[i])
            elif condition3:
                pass  # Skip this iteration, effectively removing the element in new_seq
            else:
                new_seq.append(aligned_ori_seq[i])
        else:
            if aligned_ori_seq[i] != '-' and aligned_ori_seq[i] == aligned_seq[i]:
                new_seq.append(aligned_ori_seq[i])
            elif aligned_ori_seq[i] != '-' and aligned_seq[i] == '-':
                pass
            elif aligned_ori_seq[i] == '-' and aligned_seq[i] != '-':
                new_seq.append(aligned_seq[i])
            else:
                pass
                
                
    return ''.join(new_seq)


def get_coords_or_nan(residue, atom_name):
    return list(map(convert_float32, residue[atom_name].get_coord())) if atom_name in residue else [np.nan, np.nan, np.nan]   



def convert_float32(value):
    if isinstance(value, np.float32):
        return round(float(value), 3)
    else:
        return value

def align_sequences(seq, ori_seq, mut_seq,pre_residue, post_residue,mut_pos):
    # Step 1: Align ori_seq and mut_seq to identify the exact mutation position
    if len(seq) != len(ori_seq):
        print('Length mismatch, aligning sequences.')
        
        alignments_ori = pairwise2.align.globalxx(ori_seq, seq)
        alignments_mut = pairwise2.align.globalxx(mut_seq, seq)
        
        best_alignment_ori = alignments_ori[0]
        best_alignment_mut = alignments_mut[0]
        
        aligned_ori_seq, aligned_seq1 = best_alignment_ori[0], best_alignment_ori[1]
        new_ori_seq = recover_seq(aligned_ori_seq, aligned_seq1)
        
        aligned_mut_seq, aligned_seq2 = best_alignment_mut[0], best_alignment_mut[1]
        new_mut_seq = recover_seq(aligned_mut_seq, aligned_seq2)
        
        print(len(list(new_ori_seq)), len(list(new_mut_seq)))
        if len(list(new_ori_seq)) != len(list(new_mut_seq) or len(list(new_ori_seq)) != len(list(seq))):
            print('???????????????????????????????')

        return new_ori_seq, new_mut_seq

AA_NAME_SYM = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
            'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
            'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y', 'UNK': "X",'SEC':"X"
        }
        
get_restype = lambda res_name: AA_NAME_SYM[res_name] if res_name in AA_NAME_SYM else 'X'

sequence = []
aa = "HDRFACGQEKLMNSYTIWPVXXXXX"
id_to_aa = {}
for i in range(-1, 25):
    id_to_aa[i] = aa[i]

method = 'training'
    
def extract_data_from_hd5(ori_seq,mut_seq,label, protein_name):
    
    seq = []
    protein_name, chain_name, mut_id = protein_name.split("_")
    mut_pos = int(mut_id.split('-')[1])
    pre_residue = ori_seq[mut_pos - 1]
    post_residue = mut_seq[mut_pos + 1]
    if protein_name == "4GFV":
        print('ddddddddddddddddddddd')
    pdb_path = '/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/cif'
    cif_path = '/gaozhangyang/zyj/Continuous-Discrete-Convolution/data/cif'
    cif_url = f"https://files.rcsb.org/download/{protein_name}.cif"
    pdb_url = f"https://files.rcsb.org/download/{protein_name}.pdb"

    cif_file_path = os.path.join(cif_path, f"{protein_name}.cif")
    pdb_file_path = os.path.join(pdb_path, f"{protein_name}.pdb")
    
    if os.path.exists(cif_file_path):
        # print(f"{protein_name}.cif already exists, skipping download.")
        # parser = MMCIFParser()
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
                    res_num.append(res_idx)
                    seq.append(one_letter_code)
        break
    seq = ''.join(seq)
    if len(seq) != len(ori_seq):

        new_ori_seq, new_mut_seq = align_sequences(seq, ori_seq, mut_seq,pre_residue, post_residue,mut_pos)

        ori_seq = new_ori_seq
        mut_seq = new_mut_seq


        print(f"New ori_seq: {ori_seq}")
        print(f"New mut_seq: {mut_seq}")
            
        print('error')
        print(protein_name)
        print(chain_name)
        print(mut_id)
        print(len(seq))
        print(len(ori_seq))
        print(seq)
        print(ori_seq)
# Load the fasta file.
seqs = []
names = []
dict_list = []


import csv
protein_seq_map = {}
tasks = []            
corrs_label = {}                        
mut_info = []
ori_seq = []
mut_seq = []
ddG_list = []
mut_info_list = []
with open('/gaozhangyang/zyj/evidence/VQProteinFormer-master/Prosdata/dataset/s669_test.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    # Define the columns to extract
    label_to_extract = [3,6]
    info_to_extract = [9,10,12]
    sequence_to_extract = [9]
    list_to_dump = []
    # Loop through each row in the CSV file
    for row in reader:
        # Extract the desired columns and convert them to strings
        ori_seq_extracted = row[1]
        mut_seq_extracted = row[2]
        pdb_extracted = row[3]
        mut_info = row[4]
        label_extracted = [row[i] for i in label_to_extract]
        label_extracted = '_'.join(label_extracted)
        info_extracted = [row[i] for i in info_to_extract]
        info_extracted = '-'.join(info_extracted)
        label_extracted = label_extracted + '_' + info_extracted
        ddG_extracted = row[15]
        # Do something with the extracted columns
        # str1 = ''
        # for i in range(len(ori_seq_extracted)):
        corrs_label[label_extracted] = ddG_extracted
        ori_seq.append(ori_seq_extracted)
        mut_seq.append(mut_seq_extracted)
        ddG_list.append(ddG_extracted)
        mut_info_list.append(label_extracted)
        




            

tasks_labels = list(zip(ori_seq,mut_seq,ddG_list,mut_info_list))

batch = pmap_multi(extract_data_from_hd5, tasks_labels)

with open(f"/gaozhangyang/zyj/evidence/VQProteinFormer-master/ddG_training/dataset/test_669.jsonl", "w") as file:
    for dictionary in batch:
        json.dump(dictionary, file)
        file.write('\n')

