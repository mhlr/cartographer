import os
import glob

from multiprocessing import Pool

from pdfminer.high_level import extract_text

def get_all_pdf_text_concatenated(dir_with_pdfs):
    pool = Pool(processes=None)  # use all cores
    pdf_paths = glob.glob(os.path.join(os.path.expanduser(dir_with_pdfs), "*.pdf"))
    all_papers = pool.map(extract_text, pdf_paths)
    return "\n".join(all_papers)
