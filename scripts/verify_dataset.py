#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import wfdb

class Tee:
    """Sdoppia l'output: scrive contemporaneamente in console e in un file."""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def verify_structure(data_path):
    """Verifica la presenza di file header e dati."""
    path = Path(data_path)
    hea_files = list(path.glob("**/*.hea"))
    dat_files = list(path.glob("**/*.dat"))
    
    print(f"\n🔍 Analisi cartella: {path}")
    print(f"  - File Header (.hea) trovati: {len(hea_files)}")
    print(f"  - File Segnale (.dat) trovati: {len(dat_files)}")
    
    if len(hea_files) == 0:
        print("❌ Errore: Nessun file .hea trovato. Verifica il percorso.")
        return None
    return hea_files

def analyze_sample_quality(hea_files):
    """Estrae info tecniche dai file header (sampling rate, lead, durata)."""
    print("\n📊 Estrazione parametri tecnici (campione di 10 record)...")
    
    sampling_rates = []
    num_leads = []
    durations = []
    
    sample_size = min(10, len(hea_files))
    for i in range(sample_size):
        # CORREZIONE: hea_files[i] è già un oggetto Path, usiamo directly .with_suffix('')
        # e poi convertiamo in stringa solo per la funzione wfdb.rdheader
        record_path = hea_files[i].with_suffix('')
        header = wfdb.rdheader(str(record_path))
        
        sampling_rates.append(header.fs)
        num_leads.append(header.n_sig)
        durations.append(header.sig_len / header.fs)
    
    print(f"  - Frequenza di campionamento (fs): {np.unique(sampling_rates)} Hz")
    print(f"  - Numero di Lead: {np.unique(num_leads)}")
    print(f"  - Durata media segnale: {np.mean(durations):.1f} secondi")

def check_csv_metadata(data_path):
    """Cerca e analizza file CSV di metadati."""
    path = Path(data_path)
    csv_files = list(path.glob("*.csv"))
    if csv_files:
        print(f"\n📄 Metadati CSV trovati: {[f.name for f in csv_files]}")
        for csv in csv_files:
            try:
                df = pd.read_csv(csv)
                print(f"  - {csv.name}: {len(df)} righe rilevate.")
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="Verifica dataset PhysioNet/WFDB")
    parser.add_argument("path", help="Percorso della cartella dataset (es. data/raw/ptb-xl)")
    args = parser.parse_args()

    data_path = Path(args.path)
    if not data_path.exists():
        print(f"❌ Il percorso {data_path} non esiste.")
        sys.exit(1)

    log_file_path = data_path / "info.txt"
    
    # Context manager per gestire il salvataggio automatico su info.txt
    with open(log_file_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, f)
        
        try:
            # Chiamata alla funzione verify_structure (ora definita correttamente sopra)
            hea_files = verify_structure(data_path)
            
            if hea_files:
                analyze_sample_quality(hea_files)
                check_csv_metadata(data_path)
                print("\n✅ Verifica completata.")
                print(f"📄 Log salvato in: {log_file_path}")
            else:
                sys.exit(1)
        finally:
            # Ripristina sempre lo stdout originale anche in caso di errori
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()