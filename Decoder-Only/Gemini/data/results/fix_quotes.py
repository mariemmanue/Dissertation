import glob
import csv

# Find all prediction files
files = glob.glob('*predictions.csv')
print(f"Found {len(files)} file(s).")

for filename in files:
    output_filename = filename.replace('.csv', '_fixed.csv')
    print(f"Fixing {filename}...")
    
    cleaned_lines = []
    
    with open(filename, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # THE FIX:
            # Check if the line is "Double Wrapped" (Starts AND ends with quotes)
            # Example: "3,""Text"",0"
            if line.startswith('"') and line.endswith('"'):
                # 1. Peel off the first and last quote
                # 2. Un-escape the inner quotes ("" becomes ")
                fixed_line = line[1:-1].replace('""', '"')
                
                # Safety Check: Does fixing it reveal more columns?
                # If the original was 1 column, and the fixed version is >1, then apply the fix.
                original_cols = next(csv.reader([line]))
                new_cols = next(csv.reader([fixed_line]))
                
                if len(new_cols) > len(original_cols):
                    cleaned_lines.append(fixed_line)
                else:
                    # It was a valid quoted line (unlikely for a whole row), keep as is
                    cleaned_lines.append(line)
            else:
                # Normal row (like row 0, 1, 2)
                cleaned_lines.append(line)
                
    # Save the repaired file
    with open(output_filename, 'w', encoding='utf-8', newline='') as f_out:
        f_out.write('\n'.join(cleaned_lines))
        
    print(f" -> Saved as {output_filename}")
