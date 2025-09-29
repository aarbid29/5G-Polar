

def convert_column_to_row(input_file, output_file):
   
    with open(input_file, 'r') as f:
      
        numbers = [line.strip() for line in f if line.strip()]

    
    row_format = ",".join(numbers)

    with open(output_file, 'w') as f:
        f.write(row_format)

    print(f"Converted {len(numbers)} numbers into row format.")
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
 
    input_file = "input.txt"   
    output_file = "output.txt" 

    convert_column_to_row(input_file, output_file)
