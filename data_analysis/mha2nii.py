import os
import itk

def convert_mha_to_nifti(input_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    # List all MHA files in the input directory
    mha_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.mha')]

    for mha_file in mha_files:
        
        input_path = os.path.join(input_dir, mha_file)
        image = itk.imread(input_path)

        output_filename = os.path.splitext(mha_file)[0] + ".nii.gz"
        output_path = os.path.join(output_dir, output_filename)

        itk.imwrite(image, output_path, compression=True)

        print(f"Converted {mha_file} to {output_filename}")

    print("Conversion completed for directory:", input_dir)

# input and output directories for images
image_input_dir = "/home/shared_data/ACOUSLIC_data/original/images/stacked_fetal_ultrasound/"
image_output_dir = "/home/hadeel/ACUSLIC_chal/data_analysis/ACOUSLIC_nifty/images/"

# input and output directories for masks
mask_input_dir = "/home/shared_data/ACOUSLIC_data/original/masks/stacked_fetal_abdomen/"
mask_output_dir = "/home/hadeel/ACUSLIC_chal/data_analysis/ACOUSLIC_nifty/masks/"

# Convert images
convert_mha_to_nifti(image_input_dir, image_output_dir)

# Convert masks
convert_mha_to_nifti(mask_input_dir, mask_output_dir)