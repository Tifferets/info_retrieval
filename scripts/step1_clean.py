"""
Step 1: Extract Text from XML Files
====================================

This script performs only text extraction from XML files.
This is the first and simplest step.
"""

# Import required libraries
import os  # Work with file system
import xml.etree.ElementTree as ET  # Process XML files
import re  # Regular expressions for text cleaning
from pathlib import Path  # Convenient work with file paths
from tqdm import tqdm  # Beautiful progress bar


def clean_text(text):
    text = re.sub(r'([^A-Za-z0-9])', r' \1 ', text)
    text = re.sub(r'\s*"\s*', r' " ', text)
    text = re.sub(r"\s*'\s*", r" ' ", text)
    text = re.sub(r'\s+', ' ', text)
    return text  # בלי strip


def extract_text_from_xml(xml_file_path):
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        all_texts = []
        skip_tags = {'gidredirect', 'publicwhip'}
        for element in root.iter():
            if element.tag in skip_tags:
                continue
            if element.text and element.text.strip():
                text = element.text.strip()
                if len(text) > 0 and not text.isspace():
                    all_texts.append(text)
            if element.tail and element.tail.strip():
                tail = element.tail.strip()
                if len(tail) > 0 and not tail.isspace():
                    all_texts.append(tail)
        if not all_texts:
            return ""
        full_text = ' '.join(all_texts)
        full_text = re.sub(r'\s+', ' ', full_text)
        full_text = full_text.strip()
        return full_text
    except ET.ParseError as e:
        print(f"⚠️  XML error in {Path(xml_file_path).name}: {e}")
        return ""
    except Exception as e:
        print(f"❌ General error in {Path(xml_file_path).name}: {e}")
        return ""


def process_xml_folder(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not input_path.exists():
        print(f"❌ Folder not found: {input_path}")
        print(f"\n💡 Tips:")
        print(f"   • Make sure the folder exists")
        print(f"   • Check the path (any typos?)")
        print(f"   • If on Windows, use \\ or /")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    xml_files = list(input_path.glob('*.xml'))

    if not xml_files:
        print(f"⚠️  No XML files found in {input_path}")
        return

    print(f"\n{'='*70}")
    print(f"🎯 Found {len(xml_files)} XML files")
    print(f"📁 Input folder: {input_path}")
    print(f"📁 Output folder: {output_path}")
    print(f"{'='*70}\n")

    success_count = 0
    empty_count = 0
    failed_count = 0
    empty_files = []

    for xml_file in tqdm(xml_files, desc="Extracting text", unit="file"):
        text = extract_text_from_xml(xml_file)
        if text:
            output_file = output_path / f"{xml_file.stem}.txt"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                success_count += 1
            except Exception as e:
                print(f"\n❌ Error saving {xml_file.name}: {e}")
                failed_count += 1
        else:
            empty_count += 1
            empty_files.append(xml_file.name)

    print(f"\n{'='*70}")
    print(f"📊 SUMMARY:")
    print(f"{'='*70}")
    print(f"  ✅ Successfully extracted: {success_count:4d} files")
    print(f"  🔗 Skipped (no content):   {empty_count:4d} files")
    if failed_count > 0:
        print(f"  ❌ Errors:                 {failed_count:4d} files")
    print(f"  📁 Files saved in: {output_path}")
    print(f"{'='*70}\n")

    if empty_count > 0:
        print(f"ℹ️  About the {empty_count} skipped files:")
        print(f"   These files contain only <gidredirect> tags (redirect pointers)")
        if len(empty_files) <= 10:
            for fname in empty_files:
                print(f"     • {fname}")
        else:
            for fname in empty_files[:10]:
                print(f"     • {fname}")
            print(f"     ... and {len(empty_files) - 10} more")
        print()

    if failed_count == 0:
        print(f"🎉 Excellent! All files with content were processed successfully!")
        print(f"💡 Next step: Clean and separate punctuation (Step 2)")
    else:
        print(f"⚠️  {failed_count} files had errors - check messages above")


### ADDED ###
def clean_all_text_files(folder_path):
    """
    Apply clean_text() to all .txt files in the given folder.
    Cleans punctuation spacing and overwrites the files.
    """
    folder = Path(folder_path)
    txt_files = list(folder.glob("*.txt"))

    if not txt_files:
        print(f"⚠️  No .txt files found in {folder}")
        return

    print(f"\n🧹 Cleaning {len(txt_files)} text files in {folder}...\n")

    for txt_file in tqdm(txt_files, desc="Cleaning text", unit="file"):
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                text = f.read()
            cleaned = clean_text(text)
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(cleaned)
        except Exception as e:
            print(f"❌ Error cleaning {txt_file.name}: {e}")

    print(f"\n✅ Finished cleaning all text files in {folder}\n")
### END ADDED ###


def preview_extraction(xml_file_path, num_chars=500):
    print(f"\n{'='*70}")
    print(f"🔍 Preview: {Path(xml_file_path).name}")
    print(f"{'='*70}\n")
    text = extract_text_from_xml(xml_file_path)
    if text:
        word_count = len(text.split())
        char_count = len(text)
        print(f"📊 Statistics:")
        print(f"  • Length: {char_count:,} characters")
        print(f"  • Words: {word_count:,} words")
        print(f"\n📝 Beginning of text (first {num_chars} characters):")
        print(f"{'-'*70}")
        print(text[:num_chars] + "...")
        print(f"{'-'*70}\n")
        print(f"💡 Looks good?")
        return True
    else:
        print("❌ Failed to extract text from file")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Step 1: Extract Text from XML Files                  ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print("1️⃣  Quick check - Preview of one file")
    print("2️⃣  Full processing - Extract text from all files")
    print("3️⃣  Custom - Set your own paths\n")
    choice = input("Choose option (1/2/3): ").strip()

    if choice == '1':
        folder = input("\nFolder with XML files: ").strip()
        xml_files = list(Path(folder).glob('*.xml'))
        if xml_files:
            print(f"\n✅ Found {len(xml_files)} XML files")
            print(f"📄 Showing first file: {xml_files[0].name}")
            preview_extraction(xml_files[0])
        else:
            print(f"\n❌ No XML files found in {folder}")

    elif choice == '2':
        default_input = "parliament_data/raw_xml"
        default_output = "parliament_data/extracted_text"
        input_folder = input(f"\nInput folder [{default_input}]: ").strip() or default_input
        output_folder = input(f"Output folder [{default_output}]: ").strip() or default_output
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm in ('y', 'yes'):
            process_xml_folder(input_folder, output_folder)
            ### ADDED ###
            ### END ADDED ###
        else:
            print("\n❌ Cancelled by user")

    elif choice == '3':
        input_folder = input("\nInput folder (XML): ").strip()
        output_folder = input("Output folder (TXT): ").strip()
        process_xml_folder(input_folder, output_folder)
        ### ADDED ###
        clean_all_text_files(output_folder)
        ### END ADDED ###
    else:
        print("\n❌ Invalid choice")


if __name__ == "__main__":
    main()
