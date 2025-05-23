# Data Access

The `.npy` files in  within the `connectivity_data` folder are averaged connectivity matrices extracted from [Domhof et al. (2022)](http://doi.org/10.25493/NVS8-XS5).

If you remove them, you can regenerate the data by following these steps:

1. Go to the [original dataset page](http://doi.org/10.25493/NVS8-XS5)
2. Navigate to `Get Data`
3. Select and download `294-Julich-Brain.zip` file
4. Once downloaded, unzip the file
5. Rename the folder `1StructuralConnectivity` to `structural_connectivity_data` and the folder `2FunctionalConnectivity` to `functional_connectivity_data`
6. Move these two folders to the `julich_brain_data` folder in the root directory of the `macro-eeg-model` project

After following these steps, rerunning the simulation should (re)generate the `.npy` files in the `connectivity_data` folder.