import os
import re

directory = "/Users/philprice/freqtrade/user_data/strategies/SimpleStrategies"

# Pattern to find buy_params and sell_params assignment blocks
# We look for the assignment and then the opening brace
buy_pattern = re.compile(r'(buy_params = \{)(\s*)', re.MULTILINE)
sell_pattern = re.compile(r'(sell_params = \{)(\s*)', re.MULTILINE)

# Pattern to find the class inheritance
class_pattern = re.compile(r'class \w+\((?P<parent>\w+)\):')

for filename in os.listdir(directory):
    if not filename.endswith(".py") or filename == "SimpleStrategy.py":
        continue
    
    path = os.path.join(directory, filename)
    with open(path, 'r') as f:
        content = f.read()
    
    # Identify parent class
    match = class_pattern.search(content)
    if not match:
        continue
    parent = match.group('parent')
    
    # We only update if it inherits from SimpleStrategy or EhlersBase or Ehlers_Indicator
    if parent not in ["SimpleStrategy", "EhlersBase", "Ehlers_Indicator"]:
        continue
    
    new_content = content
    
    # Update buy_params if not already inherited
    if f"**{parent}.buy_params" not in content:
        # Replace the opening of the dict with the parent spread
        new_content = buy_pattern.sub(fr'\1\2**{parent}.buy_params,\2', new_content)
    
    # Update sell_params if not already inherited
    if f"**{parent}.sell_params" not in new_content:
        new_content = sell_pattern.sub(fr'\1\2**{parent}.sell_params,\2', new_content)
        
    if new_content != content:
        print(f"Updating {filename} (parent: {parent})")
        with open(path, 'w') as f:
            f.write(new_content)
