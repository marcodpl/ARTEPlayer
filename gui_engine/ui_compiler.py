import os
from pathlib import Path
if __name__ == "__main__":
    while True:
        sel = input("Instruct: ")
        if "cl " in sel:
            if sel == "cl all":
                paths = list( Path("./guis").glob("*.ui") )
                for path in paths:
                    if os.path.exists(str(path).split(".")[0] + ".py"):
                        pass
                    else:
                        os.system(f"pyuic5 {str(path)} -o {str(path).split('.')[0]}.py")
            else:
                path = sel.split(" ")[1]
                assert os.path.exists(path)
                os.system(f"pyuic5 {path} -o {path.split('.')[0]}.py")
        elif "clean" in sel:
            paths = list( Path("./guis").glob("*.ui") )
            for path in paths:
                if os.path.exists(str(path).split(".")[0] + ".py"):
                    os.remove(str(path).split(".")[0] + ".py")