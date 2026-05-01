from Bio import Entrez, SeqIO

def download_chloroplast():
    Entrez.email = "663vova@gmail.com"  
    handle = Entrez.efetch(db="nucleotide", id="NC_000932.1", rettype="fasta", retmode="text")
    record = SeqIO.read(handle, "fasta")
    handle.close()
    
    with open("data/negatives/chloroplast.fasta", "w") as f:
        SeqIO.write(record, f, "fasta")
    print(f"Скачано: {record.description}")

download_chloroplast()