import re
pattern = r"^\d+\w+\/\w+\d\/\w+\/"
string = "4GV/X13/S/789/D/"
match = re.match(pattern, string)
if match:
    print(match.group(0).split("/"))
    