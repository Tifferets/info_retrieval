"""
Step 2: Batch Lemmatization
============================

Lemmatize all text files in a folder using SpaCy.

This script:
1. Loads all .txt files from input folder
2. Lemmatizes each file (converts words to their root form)
3. Saves lemmatized versions to output folder

Example:
- Input: "The cats are running quickly"
- Output: "the cat be run quickly"
"""

import spacy
from pathlib import Path
from tqdm import tqdm
import time


def lemmatize_folder(input_folder, output_folder, batch_size=100):
    """
    Lemmatize all text files in a folder
    
    Args:
        input_folder (str): Path to folder with clean text files
        output_folder (str): Path to folder for lemmatized files
        batch_size (int): Number of files to process before reporting (default: 100)
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Check input folder exists
    if not input_path.exists():
        print(f"❌ Error: Input folder does not exist: {input_path}")
        return
    
    # Create output folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load SpaCy model
    print("\n🔄 Loading SpaCy model (en_core_web_sm)...")
    print("   (This may take a few seconds...)")
    
    try:
        # Disable unnecessary components for speed
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        print("✅ SpaCy model loaded successfully!")
    except Exception as e:
        print(f"\n❌ Error loading SpaCy model: {e}")
        print("\n💡 You may need to install it:")
        print("   python -m spacy download en_core_web_sm")
        return
    
    # Find all text files
    txt_files = sorted(list(input_path.glob('*.txt')))
    
    if not txt_files:
        print(f"\n❌ No .txt files found in {input_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"📋 Configuration:")
    print(f"{'='*70}")
    print(f"Input folder:  {input_path}")
    print(f"Output folder: {output_path}")
    print(f"Files found:   {len(txt_files)}")
    print(f"{'='*70}\n")
    
    # Statistics
    success_count = 0
    failed_count = 0
    total_time = 0
    
    # Process each file
    print("🔄 Processing files...\n")
    
    for txt_file in tqdm(txt_files, desc="Lemmatizing", unit="file"):
        start_time = time.time()
        
        try:
            # Read input file
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Skip empty files
            if not text.strip():
                failed_count += 1
                continue
            
            # Process with SpaCy
            doc = nlp(text)
            
            # Extract lemmas
            lemmatized_text = " ".join([token.lemma_ for token in doc])
            
            # Save to output folder (same filename)
            output_file = output_path / txt_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(lemmatized_text)
            
            success_count += 1
            total_time += (time.time() - start_time)
            
        except Exception as e:
            print(f"\n⚠️  Error processing {txt_file.name}: {e}")
            failed_count += 1
    
    # Calculate statistics
    avg_time = total_time / success_count if success_count > 0 else 0
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY:")
    print(f"{'='*70}")
    print(f"✅ Successfully processed: {success_count}/{len(txt_files)} files")
    
    if failed_count > 0:
        print(f"❌ Failed:                 {failed_count}/{len(txt_files)} files")
    
    print(f"⏱️  Total time:             {total_time:.1f} seconds")
    print(f"⚡ Average per file:        {avg_time:.2f} seconds")
    print(f"📁 Output saved in:        {output_path}")
    print(f"{'='*70}\n")
    
    if success_count == len(txt_files):
        print("🎉 All files lemmatized successfully!")
    
    print("\n💡 Next step: Build TF-IDF matrix with lemmatized files")


def lemmatize_single_file(input_file, output_file=None):
    """
    Lemmatize a single file (original functionality)
    
    Args:
        input_file (str): Path to input text file
        output_file (str): Path to output file (optional, auto-generated if not provided)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ Error: File does not exist: {input_path}")
        return
    
    # Auto-generate output filename if not provided
    if output_file is None:
        output_path = input_path.parent / f"{input_path.stem}_lemmatized.txt"
    else:
        output_path = Path(output_file)
    
    # Load SpaCy
    print("🔄 Loading SpaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        print("✅ SpaCy model loaded!")
    except Exception as e:
        print(f"❌ Error loading SpaCy model: {e}")
        print("\n💡 Install with: python -m spacy download en_core_web_sm")
        return
    
    # Process file
    print(f"🔄 Lemmatizing: {input_path.name}")
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        doc = nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(lemmatized_text)
        
        print(f"✅ Lemmatization complete!")
        print(f"📁 Output: {output_path}")
        
    except Exception as e:
        print(f"❌ Error processing file: {e}")


def preview_lemmatization(input_folder, num_samples=5):
    """
    Preview lemmatization on one file before processing all
    
    Args:
        input_folder (str): Path to folder with text files
        num_samples (int): Number of example sentences to show
    """
    input_path = Path(input_folder)
    
    if not input_path.exists():
        print(f"❌ Folder not found: {input_path}")
        return
    
    txt_files = list(input_path.glob('*.txt'))
    if not txt_files:
        print(f"❌ No .txt files found in {input_path}")
        return
    
    # Load SpaCy
    print("🔄 Loading SpaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Install with: python -m spacy download en_core_web_sm")
        return
    
    # Process first file
    sample_file = txt_files[0]
    print(f"\n{'='*70}")
    print(f"🔍 PREVIEW: {sample_file.name}")
    print(f"{'='*70}\n")
    
    try:
        with open(sample_file, 'r', encoding='utf-8') as f:
            text = f.read()[:1000]  # First 1000 characters
        
        doc = nlp(text)
        
        # Show examples
        print("📝 Original → Lemmatized examples:\n")
        
        word_pairs = [(token.text, token.lemma_) for token in doc if token.is_alpha][:20]
        
        for original, lemma in word_pairs:
            if original != lemma:  # Only show changes
                print(f"   {original:15s} → {lemma}")
        
        print(f"\n{'='*70}")
        print("💡 Looks good? Run the full lemmatization!")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """
    Main function
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Step 2: Batch Lemmatization                          ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
  
    default_input = r"C:\Users\USER\Desktop\school work\Year 5\aichzur meida\clean_xml"
    default_output = r"C:\Users\USER\Desktop\school work\Year 5\aichzur meida\lemmatized_xml"
    
    print(f"\n📂 Folder Configuration:")
    print(f"{'='*70}")
    
    input_folder = input(f"Input folder (Enter for default): ").strip() or default_input
    output_folder = input(f"Output folder (Enter for default): ").strip() or default_output
    
    print(f"\n{'='*70}")
    print(f"Ready to lemmatize:")
    print(f"{'='*70}")
    print(f"Input:  {input_folder}")
    print(f"Output: {output_folder}")
    print(f"{'='*70}")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    
    if confirm in ['y', 'yes']:
        lemmatize_folder(input_folder, output_folder)
    else:
        print("\n❌ Cancelled")
    
   


if __name__ == "__main__":
    main()