import json

def extract_code_markdown(notebook_file, output_filename="notebook_content.txt"):  # Added output_filename
    with open(notebook_file, 'r') as f:
        data = json.load(f)

    with open(output_filename, 'w') as outfile:  # Open the output file
        for cell in data['cells']:
            if cell['cell_type'] == 'code':
                outfile.write("**CODE BLOCK**\n")
                outfile.writelines(cell['source'])  # Write code lines directly
                outfile.write("\n")  # Add a newline between cells
            elif cell['cell_type'] == 'markdown':
                outfile.write("**MARKDOWN BLOCK**\n")
                outfile.writelines(cell['source'])
                outfile.write("\n")

 

if __name__=='__main__':
    extract_code_markdown('test2.ipynb')