import pickle
import subprocess
import os

import torch
import numpy as np


def load_from_baseline(fname):
    with open(fname, 'rb') as f:
        p = pickle.load(f)
        p = torch.tensor(p)
        indices = p[:,:,0]
        scores = p[:,:,1]
        return indices.to(torch.int64),scores


def load(fname):
    with open(fname, 'rb') as f:
        p = pickle.load(f)
        return p


def load_from_file(filename, k=10):
    with open(filename, 'rb') as f:
        indices,scores = pickle.load(f)
        return indices.to(device="cpu",dtype=torch.int64),scores.cpu()


def get_score_data(dataset, method, k=10):
    parent_path = "pickles/results"
    bert_path = "pickles/results/BERT"
    bert_inner_path = "pickles/results/BERT/colbertv2-plaid"

    method_file_map = {
        'submodlib lazy': f'greedy_submodlib_LazyGreedy_k{k}_{dataset}_bf_k{k}_submodlib_no_stop.pkl',
        'submodlib stochastic 0.5': f'greedy_submodlib_StochasticGreedy_k{k}_{dataset}_bf_k{k}_submodlib_no_stop_eps0.5.pkl',
        'submodlib ltl 0.1': f'greedy_submodlib_LazierThanLazyGreedy_k{k}_{dataset}_bf_k{k}_submodlib_no_stop.pkl',
        'submodlib ltl 0.5': f'greedy_submodlib_LazierThanLazyGreedy_k{k}_{dataset}_bf_k{k}_submodlib_no_stop_eps0.5.pkl',
        'submodlib ltl 0.9': f'greedy_submodlib_LazierThanLazyGreedy_k{k}_{dataset}_bf_k{k}_submodlib_no_stop_eps0.9.pkl',
        'exact greedy': f'greedy_base_0_128_k{k}_{dataset}_bf.pkl',
        'WARP iid': f'xtr_colbertv2-plaid_{dataset}_k{k}_xtr-base-en.pkl',
        'MUVERA iid': f'muvera_iid_{dataset}_k{k}.pkl',
        'ColBERT iid': f'norm_base_n2_d128_{dataset}_k10.pkl',
        'ColBERT angiogram - 1': f'dblnorm_aug_n2_d128_rh8_threshold1_{dataset}_k{k}_rerankperh1.pkl',
        'ColBERT angiogram - 10': f'dblnorm_aug_n2_d128_rh8_threshold1_{dataset}_k{k}_rerankperh10.pkl',
        'ColBERT angiogram - 15': f'dblnorm_aug_n2_d128_rh8_threshold1_{dataset}_k{k}_rerankperh15.pkl',
        'ColBERT angiogram - 20': f'dblnorm_aug_n2_d128_rh8_threshold1_{dataset}_k{k}_rerankperh20.pkl',
        'ColBERT bypass - 1': f'dblnorm_int_n2_d128_rh8_intTrue_extTrue_{dataset}_k{k}_rerankperh1.pkl',
        'ColBERT bypass - 10': f'dblnorm_int_n2_d128_rh8_intTrue_extTrue_{dataset}_k{k}_rerankperh10.pkl',
        'ColBERT bypass - 15': f'dblnorm_int_n2_d128_rh8_intTrue_extTrue_{dataset}_k{k}_rerankperh15.pkl',
    }

    method_path_map = {
        'submodlib lazy': parent_path,
        'submodlib stochastic 0.5': parent_path,
        'submodlib ltl 0.1': parent_path,
        'submodlib ltl 0.5': parent_path,
        'submodlib ltl 0.9': parent_path,
        'exact greedy': parent_path,
        'WARP iid': parent_path,
        'MUVERA iid': bert_path,
        'ColBERT iid': bert_inner_path,
        'ColBERT angiogram - 1': bert_inner_path,
        'ColBERT angiogram - 10': bert_inner_path,
        'ColBERT angiogram - 15': bert_inner_path,
        'ColBERT angiogram - 20': bert_inner_path,
        'ColBERT bypass - 1': bert_inner_path,
        'ColBERT bypass - 10': bert_inner_path,
        'ColBERT bypass - 15': bert_inner_path,
    }

    filename = method_file_map[method]
    path = os.path.join(os.getcwd(), method_path_map[method])
    print(f"Method is {method}, loading from {os.path.join(path, filename)}")

    try:
        if method == "exact greedy":
            inds, scores = load_from_baseline(os.path.join(path, filename))
        else:
            inds, scores = load(os.path.join(path, filename))
    except:
        print(f"File not found: {filename}")
        print(f"{method} + {dataset} does not have k={k} data, defaulting to k=15")
        filename = filename.replace(f'k{k}', 'k15')

        try:
            if method == "exact greedy":
                inds, scores = load_from_baseline(os.path.join(path, filename))
            else:
                inds, scores = load(os.path.join(path, filename))

            inds = inds[:k]
            scores = scores[:k]
        except FileNotFoundError:
            raise FileNotFoundError(f"k=15 file not found either: {os.path.join(path, filename)}")
        except Exception as e:
            raise e

    return inds, scores


def crop_pdf_with_pdfcrop(input_path, output_path=None):
    """
    Crop PDF using pdfcrop (requires texlive-extra-utils)
    This is the most reliable method for LaTeX-generated PDFs
    """
    if output_path is None:
        output_path = input_path.replace('.pdf', '_cropped.pdf')
    
    try:
        subprocess.run(['pdfcrop', input_path, output_path], 
                      check=True, capture_output=True)
        print(f"Successfully cropped {input_path} -> {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"pdfcrop failed: {e}")
        return None
    except FileNotFoundError:
        print("pdfcrop not found. Install with: sudo apt-get install texlive-extra-utils")
        return None

def crop_pdf_with_pypdf(input_path, output_path=None, margin=10):
    """
    Crop PDF using PyPDF2/PyPDF4 - pure Python solution
    margin: points to keep around content (72 points = 1 inch)
    """
    try:
        from PyPDF2 import PdfReader, PdfWriter
        from PyPDF2.generic import RectangleObject
    except ImportError:
        try:
            from PyPDF4 import PdfFileReader as PdfReader, PdfFileWriter as PdfWriter
            from PyPDF4.generic import RectangleObject
        except ImportError:
            print("Neither PyPDF2 nor PyPDF4 found. Install with: pip install PyPDF2")
            return None
    
    if output_path is None:
        output_path = input_path.replace('.pdf', '_cropped.pdf')
    
    try:
        with open(input_path, 'rb') as file:
            reader = PdfReader(file)
            writer = PdfWriter()
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                
                # Get the bounding box of the content
                bbox = page.mediabox
                
                # Create a new cropped page (this is basic - you might need to adjust)
                page.cropbox = RectangleObject([
                    bbox.lower_left[0] + margin,
                    bbox.lower_left[1] + margin, 
                    bbox.upper_right[0] - margin,
                    bbox.upper_right[1] - margin
                ])
                
                writer.add_page(page)
            
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
                
        print(f"Successfully cropped {input_path} -> {output_path}")
        return output_path
        
    except Exception as e:
        print(f"PyPDF cropping failed: {e}")
        return None


def crop_pdf_with_fitz(input_path, output_path=None, margin=5):
    """
    Crop PDF using PyMuPDF (fitz) - most advanced option
    Automatically detects content boundaries
    margin: points to keep around content
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("PyMuPDF not found. Install with: pip install PyMuPDF")
        return None
    
    if output_path is None:
        output_path = input_path.replace('.pdf', '_cropped.pdf')
    
    try:
        doc = fitz.open(input_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get the actual content bounding box (ignoring whitespace)
            bbox = page.get_text("dict")
            if bbox and "blocks" in bbox:
                # Find the actual content boundaries
                min_x = float('inf')
                min_y = float('inf') 
                max_x = float('-inf')
                max_y = float('-inf')
                
                for block in bbox["blocks"]:
                    if "lines" in block:  # Text block
                        for line in block["lines"]:
                            for span in line["spans"]:
                                rect = span["bbox"]
                                min_x = min(min_x, rect[0])
                                min_y = min(min_y, rect[1])
                                max_x = max(max_x, rect[2])
                                max_y = max(max_y, rect[3])
                    else:  # Image block
                        rect = block["bbox"]
                        min_x = min(min_x, rect[0])
                        min_y = min(min_y, rect[1])
                        max_x = max(max_x, rect[2])
                        max_y = max(max_y, rect[3])
                
                if min_x != float('inf'):
                    # Apply margin and set crop box
                    crop_rect = fitz.Rect(
                        max(0, min_x - margin),
                        max(0, min_y - margin),
                        min(page.rect.width, max_x + margin),
                        min(page.rect.height, max_y + margin)
                    )
                    page.set_cropbox(crop_rect)
        
        doc.save(output_path)
        doc.close()
        
        print(f"Successfully cropped {input_path} -> {output_path}")
        return output_path
        
    except Exception as e:
        print(f"PyMuPDF cropping failed: {e}")
        return None