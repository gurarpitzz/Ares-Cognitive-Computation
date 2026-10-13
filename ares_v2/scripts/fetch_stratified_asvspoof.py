import io
import time
from pathlib import Path
import pyarrow.parquet as pq
import fsspec
import soundfile as sf
import torch

def fetch_stratified():
    url = 'https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA/resolve/main/data/validation-00000-of-00001.parquet'
    fs, path = fsspec.core.url_to_fs(url)
    
    # We want balanced bonafide and spoof across all attacks and multiple speakers
    # RG 0, 5, 15: Bonafide (LA_0069, LA_0072, LA_0073, LA_0099, LA_0100) -> 300 bonafide
    # RG 30: A01 (LA_0070) -> 60 samples
    # RG 100: A03 (LA_0069) -> 60 samples
    # RG 150: A04 (LA_0072) -> 60 samples
    # RG 200: A05 (LA_0076) -> 60 samples
    # RG 240: A06 (LA_0076, LA_0077) -> 60 samples
    # Total: 300 bonafide, 300 spoof = 600 total across 6 attacks and 8+ speakers!
    
    rgs_to_fetch = [
        (0, 100, "bonafide"),
        (5, 100, "bonafide"),
        (15, 100, "bonafide"),
        (30, 60, "spoof_A01"),
        (100, 60, "spoof_A03"),
        (150, 60, "spoof_A04"),
        (200, 60, "spoof_A05"),
        (240, 60, "spoof_A06"),
    ]
    
    all_samples = []
    
    for rg, max_n, desc in rgs_to_fetch:
        t0 = time.time()
        print(f"Reading Row Group {rg} ({desc})...", flush=True)
        # Re-open file handle for each RG to ensure clean HTTP range connection
        with fs.open(path) as f_handle:
            pf = pq.ParquetFile(f_handle)
            table = pf.read_row_group(rg)
            df = table.to_pandas()
            
        print(f"  Downloaded RG {rg} in {time.time()-t0:.2f}s, parsing {min(max_n, len(df))} rows...", flush=True)
        sub_df = df.iloc[:max_n]
        
        for _, row in sub_df.iterrows():
            audio_dict = row["audio"]
            audio_bytes = audio_dict["bytes"]
            data, sr = sf.read(io.BytesIO(audio_bytes))
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype('float32')
            
            all_samples.append({
                "speaker_id": str(row["speaker_id"]),
                "audio_file_name": str(row["audio_file_name"]),
                "system_id": str(row["system_id"]),
                "key": int(row["key"]),
                "waveform": data,
                "sample_rate": sr,
            })
            
        print(f"  Cumulative samples: {len(all_samples)}", flush=True)
        
    out_path = Path("data/processed/asvspoof_stratified_cache.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_samples, out_path)
    print(f"\nSUCCESS: Saved {len(all_samples)} stratified samples to {out_path}!", flush=True)
    
    # Print speaker and system distributions
    import collections
    bonafide_spk = collections.Counter(s["speaker_id"] for s in all_samples if s["key"] == 0)
    spoof_spk = collections.Counter(s["speaker_id"] for s in all_samples if s["key"] == 1)
    spoof_sys = collections.Counter(s["system_id"] for s in all_samples if s["key"] == 1)
    print("Bonafide Speakers:", bonafide_spk, flush=True)
    print("Spoof Speakers:", spoof_spk, flush=True)
    print("Spoof Systems:", spoof_sys, flush=True)

if __name__ == "__main__":
    fetch_stratified()
