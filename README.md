# jflab-minipipe

The JF-Lab miniature microscope pipeline for preprocessing and analysis.

1. Get your .mkv files to be merged, downsampled and/or motion corrected.
2. Run the following on the .mkv files:  
  `python minipipe.py file1.mkv file2.mkv -c 2000 --motion_corr --cores 8`  
  
  Flags:  
  -d/--downsample: downsample factor  
  -c/--chunk_size: chunk_size  
  --motion_corr: if you want to motion correct_motion  
  -t/--threshold: if you want to indicate threshold  
  -target_frame: if you want to choose a frame other than the first to reference  
  --cores: number of threads to run in parallel  
  --bigtiff: If .mkv(s) amount to > 12Gb, must use this mode or memory error will occur  
  --merge: merge all the files instead of individually processing them  
  -o/--output: If --merge, then the name for the merged .tiff file  
3. Run [CNMF-E](https://github.com/zhoupc/CNMF_E) on the .tiff files.
4. Use review_traces.py to manually inspect the neurons to keep or exclude from analysis.
