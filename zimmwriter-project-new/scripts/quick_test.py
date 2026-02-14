"""Quick test: connect to ZimmWriter and print status."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controller import ZimmWriterController

zw = ZimmWriterController()
if zw.connect():
    print(f"✅ Connected: {zw.get_window_title()}")
    print(f"   Buttons: {len(zw.get_all_buttons())}")
    print(f"   Checkboxes: {len(zw.get_all_checkboxes())}")
    print(f"   Dropdowns: {len(zw.get_all_dropdowns())}")
    print(f"   Text fields: {len(zw.get_all_text_fields())}")
else:
    print("❌ ZimmWriter not found. Make sure it's running.")
    sys.exit(1)
