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


def extract_text_from_xml(xml_file_path):
    """
    Extract ALL text content from a UK Parliament XML file
    
    This function extracts ALL text from the XML file, including:
    - Speeches (in <speech> tags)
    - Names (in <mpname> tags for votes/divisions)
    - Any other text content
    
    It ignores:
    - XML tags themselves (we only want the text inside)
    - gidredirect tags (these are just redirects, no useful text)
    - Attributes (like person_id, vote="yes", etc.)
    
    Note: Some XML files contain ONLY gidredirect tags (redirect files).
    These files legitimately have no text content to extract.
    
    Process:
    1. Read and parse the XML file
    2. Recursively walk through ALL elements
    3. Extract text from every element (except gidredirect)
    4. Clean and join all text pieces
    
    Args:
        xml_file_path (str): Full path to XML file
        
    Returns:
        str: Extracted clean text, or empty string if no text found
    """
    try:
        # Step 1: Read and parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Step 2: Prepare list to collect all text pieces
        all_texts = []
        
        # Tags to skip completely (these contain no useful text)
        # gidredirect is just metadata for redirects
        skip_tags = {'gidredirect', 'publicwhip'}  # publicwhip is usually the root
        
        # Step 3: Recursively extract text from ALL elements
        # We use root.iter() which goes through every element in the tree
        for element in root.iter():
            # Skip tags that we don't want (like gidredirect)
            if element.tag in skip_tags:
                continue
            
            # Extract the direct text of the element
            # This is text that comes right after the opening tag
            # Example: <mpname>John Smith</mpname>
            #          element.text = "John Smith"
            if element.text and element.text.strip():
                text = element.text.strip()
                # Only add if it's meaningful text (not just whitespace/newlines)
                if len(text) > 0 and not text.isspace():
                    all_texts.append(text)
            
            # Extract the tail text of the element
            # This is text that comes after the closing tag
            # Example: <mpname>John</mpname> voted yes
            #          element.tail = " voted yes"
            if element.tail and element.tail.strip():
                tail = element.tail.strip()
                # Only add if it's meaningful text
                if len(tail) > 0 and not tail.isspace():
                    all_texts.append(tail)
        
        # Step 4: Check if we found anything
        if not all_texts:
            # This is normal for files that only contain gidredirect tags
            # These are "redirect" files that point to the actual debate file
            # We return empty string (not an error)
            return ""
        
        # Step 5: Join all texts into one string
        # join connects all elements in the list with spaces between them
        full_text = ' '.join(all_texts)
        
        # Step 6: Basic cleaning
        # Replace multiple whitespaces with single space
        full_text = re.sub(r'\s+', ' ', full_text)
        
        # Remove any remaining special characters that might cause issues
        # But keep basic punctuation
        full_text = full_text.strip()
        
        return full_text
    
    except ET.ParseError as e:
        # XML Parsing error - the file is invalid or corrupted
        print(f"⚠️  XML error in {Path(xml_file_path).name}: {e}")
        return ""
    
    except Exception as e:
        # Any other error we didn't anticipate
        print(f"❌ General error in {Path(xml_file_path).name}: {e}")
        return ""


def process_xml_folder(input_folder, output_folder):
    """
    Process all XML files in folder and save as text files
    
    This function loops through all XML files in a folder, extracts text
    from each file, and saves it as a .txt file with the same name.
    
    Note: Some files may contain only <gidredirect> tags (redirect files).
    These files have no text content and will be skipped - this is normal!
    
    Process:
    1. Check that input folder exists
    2. Create output folder if it doesn't exist
    3. Find all XML files
    4. Process each file and save the text
    5. Display process summary
    
    Args:
        input_folder (str): Path to folder with XML files
        output_folder (str): Path to folder for saving TXT files
        
    Returns:
        None - function saves files and prints status
    """
    # Convert to Path objects
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Step 1: Check that input folder exists
    if not input_path.exists():
        print(f"❌ Folder not found: {input_path}")
        print(f"\n💡 Tips:")
        print(f"   • Make sure the folder exists")
        print(f"   • Check the path (any typos?)")
        print(f"   • If on Windows, use \\ or /")
        print(f"     Example: C:/Users/YourName/Downloads/xml_files")
        return
    
    # Step 2: Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 3: Find all XML files in folder
    xml_files = list(input_path.glob('*.xml'))
    
    # Check: Did we find any files?
    if not xml_files:
        print(f"⚠️  No XML files found in {input_path}")
        print(f"\n💡 Tips:")
        print(f"   • Make sure there are .xml files in the folder")
        print(f"   • Maybe files are in a subfolder? glob doesn't search recursively")
        print(f"   • Try printing: list(Path('{input_path}').iterdir())")
        return
    
    # Print nice header
    print(f"\n{'='*70}")
    print(f"🎯 Found {len(xml_files)} XML files")
    print(f"📁 Input folder: {input_path}")
    print(f"📁 Output folder: {output_path}")
    print(f"{'='*70}\n")
    
    # Step 4: Count different outcomes
    success_count = 0       # Files with text extracted and saved
    empty_count = 0         # Files with no text (redirect-only files)
    failed_count = 0        # Files that had errors
    empty_files = []        # Track which files were empty
    
    # Step 5: Process each file with progress bar
    for xml_file in tqdm(xml_files, desc="Extracting text", unit="file"):
        # Extract text from current file
        text = extract_text_from_xml(xml_file)
        
        # Check: Did extraction succeed?
        if text:
            # Create path for output file
            output_file = output_path / f"{xml_file.stem}.txt"
            
            try:
                # Save text to file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                # Success! Increment counter
                success_count += 1
                
            except Exception as e:
                # Error saving (disk full? no permissions?)
                print(f"\n❌ Error saving {xml_file.name}: {e}")
                failed_count += 1
        else:
            # No text extracted - this is usually a redirect-only file
            # This is NORMAL and expected!
            empty_count += 1
            empty_files.append(xml_file.name)
    
    # Step 6: Process summary
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY:")
    print(f"{'='*70}")
    print(f"  ✅ Successfully extracted: {success_count:4d} files")
    print(f"  🔗 Skipped (no content):   {empty_count:4d} files")
    
    # If there were failures, print that too
    if failed_count > 0:
        print(f"  ❌ Errors:                 {failed_count:4d} files")
        print(f"\n💡 Tip: See error messages above for details")
    
    print(f"  📁 Files saved in: {output_path}")
    print(f"{'='*70}\n")
    
    # Explain empty files
    if empty_count > 0:
        print(f"ℹ️  About the {empty_count} skipped files:")
        print(f"   These files contain only <gidredirect> tags (redirect pointers)")
        print(f"   They have no actual text content - this is NORMAL!")
        print(f"   Example: A file might redirect to the main debate file")
        
        # Show first 10 empty files
        if len(empty_files) <= 10:
            print(f"\n   Skipped files:")
            for fname in empty_files:
                print(f"     • {fname}")
        else:
            print(f"\n   First 10 skipped files:")
            for fname in empty_files[:10]:
                print(f"     • {fname}")
            print(f"     ... and {len(empty_files) - 10} more")
        print()
    
    # Success message
    if failed_count == 0:
        print(f"🎉 Excellent! All files with content were processed successfully!")
        print(f"💡 Next step: Clean and separate punctuation (Step 2)")
    else:
        print(f"⚠️  {failed_count} files had errors - check messages above")




def preview_extraction(xml_file_path, num_chars=500):
    """
    Preview text extraction from one file
    
    Very useful function for quick testing!
    It shows you how the extracted text looks without processing all files.
    
    Use it to:
    • Check that extraction works correctly
    • See an example before processing everything
    • Identify problems early
    
    Args:
        xml_file_path (str): Path to XML file
        num_chars (int): How many characters to display (default: 500)
        
    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    # Print nice header
    print(f"\n{'='*70}")
    print(f"🔍 Preview: {Path(xml_file_path).name}")
    print(f"{'='*70}\n")
    
    # Extract text from file
    # Call the extract_text_from_xml function we wrote above
    text = extract_text_from_xml(xml_file_path)
    
    # Check: Did extraction succeed?
    if text:
        # Calculate statistics about the text
        # split() breaks text into words (splits by spaces)
        word_count = len(text.split())
        # len(text) is simply the length of the string in characters
        char_count = len(text)
        
        # Print statistics
        print(f"📊 Statistics:")
        print(f"  • Length: {char_count:,} characters")  # :, adds commas for readability
        print(f"  • Words: {word_count:,} words")
        
        # Print beginning of text
        print(f"\n📝 Beginning of text (first {num_chars} characters):")
        print(f"{'-'*70}")
        # text[:num_chars] is slicing - takes the first num_chars characters
        print(text[:num_chars] + "...")
        print(f"{'-'*70}\n")
        
        # Print helpful tip
        print(f"💡 Looks good?")
        print(f"   • If yes - run again and choose option 2 to process all files!")
        print(f"   • If no - tell me what the error is and we'll fix it")
        
        return True  # Extraction succeeded
    else:
        # Extraction failed
        print("❌ Failed to extract text from file")
        print("\n💡 Possible reasons:")
        print("   • File is empty or corrupted")
        print("   • No <speech> or <p> tags in file")
        print("   • XML parsing error (check error message above)")
        return False  # Extraction failed


def main():
    """
    Main function - entry point to the script
    
    Here you choose what to do:
    1. Preview of one file (recommended to start!)
    2. Full processing of all files
    3. Custom with your own paths
    """
    # Print nice header
    print("""
╔══════════════════════════════════════════════════════════════╗
║         Step 1: Extract Text from XML Files                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Show options
    print("You have 3 options:\n")
    print("1️⃣  Quick check - Preview of one file")
    print("    (Recommended to start! See if everything works)")
    print("\n2️⃣  Full processing - Extract text from all files")
    print("    (After you verify option 1 works)")
    print("\n3️⃣  Custom - Set your own paths")
    print("    (For advanced users)\n")
    
    # Get choice from user
    # input() waits for user to type something and press Enter
    # strip() removes extra whitespace at beginning and end
    choice = input("Choose option (1/2/3): ").strip()
    
    # ============================================
    # Option 1: Preview of one file
    # ============================================
    if choice == '1':
        print("\n" + "="*70)
        print("📂 Where are your files?")
        print("="*70)
        print("\n💡 Path examples:")
        print("   Windows:  C:/Users/YourName/Downloads/xml_files")
        print("             D:/University/NLP/parliament_data/raw_xml")
        print("   Linux:    /home/user/parliament/raw_xml")
        print("   Mac:      /Users/username/Documents/xml_files")
        
        # Get path from user
        folder = input("\nFolder with XML files: ").strip()
        
        # Find first file in folder
        # list() converts result to list
        # If there are files, xml_files won't be empty
        xml_files = list(Path(folder).glob('*.xml'))
        
        if xml_files:
            # Found files! Show the first one
            print(f"\n✅ Found {len(xml_files)} XML files")
            print(f"📄 Showing first file: {xml_files[0].name}")
            
            # Show preview
            preview_extraction(xml_files[0])
            
            # Guide what to do next
            print("\n" + "="*70)
            print("🎯 What now?")
            print("="*70)
            print("✅ If text looks correct → Run again and choose option 2")
            print("❌ If there's a problem → Tell me what the error is")
        else:
            # No files found
            print(f"\n❌ No XML files found in {folder}")
            print("\n💡 Things to check:")
            print("   • Is the path correct? (maybe a typo)")
            print("   • Are files in a subfolder?")
            print("   • Do files end with .xml?")
    
    # ============================================
    # Option 2: Full processing of all files
    # ============================================
    elif choice == '2':
        print("\n" + "="*70)
        print("📂 Path settings:")
        print("="*70)
        
        # Default paths
        # These will work if you run code in correct folder
        default_input = "parliament_data/raw_xml"
        default_output = "parliament_data/extracted_text"
        
        # Get input path
        print(f"\n1️⃣  Input folder (where XML files are):")
        print(f"   Default: {default_input}")
        input_folder = input("   Or enter different path (Enter for default): ").strip()
        # If user didn't type anything, use default
        if not input_folder:
            input_folder = default_input
        
        # Get output path
        print(f"\n2️⃣  Output folder (where to save TXT files):")
        print(f"   Default: {default_output}")
        output_folder = input("   Or enter different path (Enter for default): ").strip()
        # If user didn't type anything, use default
        if not output_folder:
            output_folder = default_output
        
        # Show summary before starting
        print(f"\n{'='*70}")
        print(f"🎯 Ready to process:")
        print(f"{'='*70}")
        print(f"  📁 Input:  {input_folder}")
        print(f"  📁 Output: {output_folder}")
        print(f"{'='*70}")
        
        # User confirmation
        # Important! To prevent mistakes
        confirm = input("\nContinue? (y/n): ").strip().lower()
        
        # Check if user confirmed
        # lower() converts to lowercase (Y -> y)
        if confirm == 'y' or confirm == 'yes':
            # Let's go, process!
            process_xml_folder(input_folder, output_folder)
            
            # Print completion message
            print("\n" + "="*70)
            print("✅ Done!")
            print("="*70)
            print("💡 What now?")
            print("   → Next step: Clean and separate punctuation")
            print("   → This will be in Step 2 (we'll do it later)")
        else:
            # User cancelled
            print("\n❌ Cancelled by user")
            print("💡 You can run again anytime!")
    
    # ============================================
    # Option 3: Custom
    # ============================================
    elif choice == '3':
        print("\n" + "="*70)
        print("📂 Enter paths:")
        print("="*70)
        
        # Get paths directly from user
        # No defaults - for advanced users!
        input_folder = input("\nInput folder (XML): ").strip()
        output_folder = input("Output folder (TXT): ").strip()
        
        # Process directly without confirmation
        # (User chose advanced option so they know what they're doing)
        process_xml_folder(input_folder, output_folder)
    
    # ============================================
    # Invalid choice
    # ============================================
    else:
        print("\n❌ Invalid choice")
        print("💡 Run again and choose 1, 2, or 3")


# ============================================
# Script entry point
# ============================================
# This code runs only when running this file directly
# (not when importing it as a module)
if __name__ == "__main__":
    main()  # Call main function