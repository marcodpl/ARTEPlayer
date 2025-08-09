import os
import sys
import importlib
import inspect
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget

def list_ui_modules():
    current = os.path.basename(__file__)
    return [f[:-3] for f in os.listdir(".") if f.endswith(".py") and f != current]

def select_module(modules):
    print("Available UI modules:")
    for i, name in enumerate(modules):
        print(f"{i + 1}. {name}")
    while True:
        try:
            index = int(input("Select a UI module by number: ")) - 1
            if 0 <= index < len(modules):
                return modules[index]
        except ValueError:
            pass
        print("Invalid selection. Try again.")

def find_ui_class(module):
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if name.startswith("Ui_"):
            return obj
    raise Exception("No class starting with 'Ui_' found.")

def main():
    app = QApplication(sys.argv)

    modules = list_ui_modules()
    if not modules:
        print("No UI modules found.")
        return

    module_name = select_module(modules)
    try:
        mod = importlib.import_module(module_name)
        UiClass = find_ui_class(mod)

        # Try to determine if the UI is meant for a main window or widget
        if "MainWindow" in UiClass.__name__:
            base = QMainWindow()
        else:
            base = QWidget()

        ui = UiClass()
        ui.setupUi(base)
        base.show()

        sys.exit(app.exec_())
    except Exception as e:
        print(f"Failed to load UI: {e}")

if __name__ == "__main__":
    main()
