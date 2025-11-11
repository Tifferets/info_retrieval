#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install requests beautifulsoup4 lxml 0- לקוריד

"""
סקריפט להורדת קבצי XML של דיוני הפרלמנט הבריטי
UK Parliament Debates XML Files Downloader

מוריד קבצים מ-debates2023-06-28d.xml ואילך
Downloads files from debates2023-06-28d.xml onwards
"""

import requests
from bs4 import BeautifulSoup
import os
import time
from datetime import datetime, timedelta
import re
from pathlib import Path

class DebatesDownloader:
    def __init__(self, output_dir="debates_xml"):
        """
        אתחול המוריד
        
        Args:
            output_dir: תיקייה לשמירת הקבצים
        """
        self.base_url = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"
        self.output_dir = output_dir
        self.start_date = datetime(2023, 6, 28)
        
        # יצירת תיקיית פלט
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 תיקיית פלט: {os.path.abspath(self.output_dir)}")
        print(f"🌐 כתובת בסיס: {self.base_url}")
        print(f"📅 תאריך התחלה: {self.start_date.strftime('%Y-%m-%d')}")
        print("-" * 60)
    
    def get_available_files(self):
        """
        קבלת רשימת כל הקבצים הזמינים מהשרת
        Gets list of all available files from the server
        """
        print("🔍 סורק את השרת לקבצים זמינים...")
        
        try:
            response = requests.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # חיפוש כל הקישורים לקבצי XML
            links = soup.find_all('a', href=True)
            xml_files = []
            
            for link in links:
                href = link['href']
                if href.endswith('.xml') and href.startswith('debates'):
                    xml_files.append(href)
            
            print(f"✅ נמצאו {len(xml_files)} קבצי XML בסך הכל")
            return sorted(xml_files)
            
        except Exception as e:
            print(f"❌ שגיאה בסריקת השרת: {e}")
            return []
    
    def parse_date_from_filename(self, filename):
        """
        חילוץ תאריך משם הקובץ
        Extract date from filename
        
        Args:
            filename: שם הקובץ (למשל: debates2023-06-28d.xml)
            
        Returns:
            datetime object או None
        """
        # פורמט: debates2023-06-28d.xml או debates2023-06-28a.xml
        match = re.search(r'debates(\d{4})-(\d{2})-(\d{2})', filename)
        if match:
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except:
                return None
        return None
    
    def filter_files_from_date(self, all_files):
        """
        סינון קבצים מהתאריך הרצוי ואילך
        Filter files from desired date onwards
        """
        filtered = []
        
        for filename in all_files:
            file_date = self.parse_date_from_filename(filename)
            if file_date and file_date >= self.start_date:
                filtered.append(filename)
        
        print(f"📋 אחרי סינון: {len(filtered)} קבצים מ-{self.start_date.strftime('%Y-%m-%d')} ואילך")
        return sorted(filtered)
    
    def download_file(self, filename, retry=3):
        """
        הורדת קובץ בודד
        Download single file
        
        Args:
            filename: שם הקובץ להורדה
            retry: מספר ניסיונות חוזרים במקרה של כשל
        """
        url = self.base_url + filename
        output_path = os.path.join(self.output_dir, filename)
        
        # בדיקה אם הקובץ כבר קיים
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            if file_size > 0:
                print(f"⏭️  {filename} כבר קיים ({file_size:,} bytes) - מדלג")
                return True
        
        for attempt in range(retry):
            try:
                print(f"⬇️  מוריד: {filename} (ניסיון {attempt + 1}/{retry})...", end=" ")
                
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # שמירת הקובץ
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                file_size = len(response.content)
                print(f"✅ הצלחה! ({file_size:,} bytes)")
                
                # המתנה קצרה בין הורדות כדי לא לעמוס על השרת
                time.sleep(0.5)
                return True
                
            except Exception as e:
                print(f"❌ שגיאה: {e}")
                if attempt < retry - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"⏳ ממתין {wait_time} שניות לפני ניסיון חוזר...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ נכשל אחרי {retry} ניסיונות")
                    return False
        
        return False
    
    def download_all(self):
        """
        הורדת כל הקבצים הרלוונטיים
        Download all relevant files
        """
        print("\n" + "="*60)
        print("🚀 מתחיל הורדת קבצים")
        print("="*60 + "\n")
        
        # קבלת רשימת קבצים
        all_files = self.get_available_files()
        if not all_files:
            print("❌ לא נמצאו קבצים. בודק אם השרת זמין...")
            return
        
        # סינון קבצים מהתאריך הרצוי
        files_to_download = self.filter_files_from_date(all_files)
        
        if not files_to_download:
            print("❌ לא נמצאו קבצים מהתאריך הרצוי")
            return
        
        print(f"\n📦 מתחיל הורדה של {len(files_to_download)} קבצים...")
        print("-" * 60 + "\n")
        
        # סטטיסטיקות
        successful = 0
        failed = 0
        start_time = time.time()
        
        # הורדת כל הקבצים
        for i, filename in enumerate(files_to_download, 1):
            print(f"[{i}/{len(files_to_download)}] ", end="")
            
            if self.download_file(filename):
                successful += 1
            else:
                failed += 1
                # שמירת רשימת קבצים שנכשלו
                with open('failed_downloads.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{filename}\n")
        
        # סיכום
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("📊 סיכום הורדה")
        print("="*60)
        print(f"✅ הצליחו: {successful}")
        print(f"❌ נכשלו: {failed}")
        print(f"⏱️  זמן כולל: {elapsed_time:.2f} שניות ({elapsed_time/60:.2f} דקות)")
        print(f"📁 הקבצים נשמרו ב: {os.path.abspath(self.output_dir)}")
        
        if failed > 0:
            print(f"\n⚠️  רשימת קבצים שנכשלו נשמרה ב-'failed_downloads.txt'")
            print("   ניתן להריץ את הסקריפט שוב כדי לנסות להוריד אותם")


def retry_failed_downloads(output_dir="debates_xml"):
    """
    ניסיון חוזר להורדת קבצים שנכשלו
    Retry downloading failed files
    """
    if not os.path.exists('failed_downloads.txt'):
        print("✅ אין קבצים שנכשלו!")
        return
    
    print("🔄 מנסה להוריד שוב קבצים שנכשלו...")
    
    downloader = DebatesDownloader(output_dir)
    
    with open('failed_downloads.txt', 'r', encoding='utf-8') as f:
        failed_files = [line.strip() for line in f if line.strip()]
    
    print(f"📋 נמצאו {len(failed_files)} קבצים שנכשלו")
    
    successful = 0
    still_failed = []
    
    for filename in failed_files:
        if downloader.download_file(filename):
            successful += 1
        else:
            still_failed.append(filename)
    
    # עדכון רשימת הכשלונות
    if still_failed:
        with open('failed_downloads.txt', 'w', encoding='utf-8') as f:
            for filename in still_failed:
                f.write(f"{filename}\n")
        print(f"\n✅ הצליחו: {successful}")
        print(f"❌ עדיין נכשלו: {len(still_failed)}")
    else:
        os.remove('failed_downloads.txt')
        print(f"\n🎉 כל הקבצים הורדו בהצלחה!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='הורדת קבצי XML של דיוני הפרלמנט הבריטי',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
דוגמאות שימוש:
  python download_debates.py                    # הורדה רגילה
  python download_debates.py --output my_data   # שינוי תיקיית פלט
  python download_debates.py --retry            # ניסיון חוזר לקבצים שנכשלו
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        default='debates_xml',
        help='תיקייה לשמירת הקבצים (ברירת מחדל: debates_xml)'
    )
    
    parser.add_argument(
        '--retry', '-r',
        action='store_true',
        help='ניסיון חוזר להוריד קבצים שנכשלו'
    )
    
    args = parser.parse_args()
    
    try:
        if args.retry:
            retry_failed_downloads(args.output)
        else:
            downloader = DebatesDownloader(args.output)
            downloader.download_all()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  ההורדה הופסקה על ידי המשתמש")
        print("💡 ניתן להריץ שוב את הסקריפט - הוא ידלג על קבצים שכבר הורדו")
    except Exception as e:
        print(f"\n❌ שגיאה כללית: {e}")
        import traceback
        traceback.print_exc()
