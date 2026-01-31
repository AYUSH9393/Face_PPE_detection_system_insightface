from PyQt6.QtWidgets import QApplication
import sys
from ppe_color_tab import PPEColorTab

app = QApplication(sys.argv)
tab = PPEColorTab(True)
print("Tab created successfully!")
print(f"Tab type: {type(tab)}")
print(f"Is admin: {tab.is_admin}")
