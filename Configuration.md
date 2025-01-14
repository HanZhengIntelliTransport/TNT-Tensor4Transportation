## Configuring a Conda Environment for TNT-Tensor4Transportation

### Prerequisites
- Ensure that Conda is installed and set up correctly.
- Verify that your Conda is updated:
  ```bash
  conda update -n base -c defaults conda
  ```

---

### **Step 1: Use an Existing Conda Environment**

If the environment already exists, follow these steps to configure it:

1. **Activate the Existing Environment**  
   Replace `<your_environment_name>` with the name of your environment:
   ```bash
   conda activate <your_environment_name>
   ```

2. **Install or Update the Required Packages**  
   Run the following commands to install or update the packages:

   ```bash
   # Conda packages
   conda install -c defaults -c pytorch python=3.12 networkx numpy pandas matplotlib zarr pytorch torchvision torchaudio

   # Pip packages
   pip install tensorly Pyomo
   ```

3. **Verify Installation**  
   Check if the packages are installed correctly:
   ```bash
   conda list
   pip list
   ```

---

### **Step 2: Configure Using a YAML File for Automation**

If you prefer automation, create a YAML file (`environment_tnt.yaml`) to define the environment configuration.

#### YAML File Content
```yaml
name: TNT-Tensor4Transportation  # Name of the environment (only descriptive if updating an existing environment)
channels:
  - defaults
  - pytorch
dependencies:
  - python=3.12
  - networkx
  - numpy
  - pandas
  - matplotlib
  - zarr
  - pip
  - pytorch::pytorch
  - pytorch::torchvision
  - pytorch::torchaudio
  - pip:
    - tensorly
    - Pyomo
```

#### Apply the Configuration to an Existing Environment
Use the following command:
```bash
conda env update --name <your_environment_name> --file environment_tnt.yaml
```

This will install and update all packages listed in the YAML file within the specified environment.

---

### **Step 3: Test the Environment**

After configuration, test that the environment works as expected:

1. **Activate the Environment**  
   ```bash
   conda activate <your_environment_name>
   ```

2. **Run a Python Script**  
   Start a Python session and test importing the required libraries:
   ```python
   import networkx
   import numpy as np
   import pandas as pd
   import matplotlib
   import zarr
   import torch
   import tensorly
   import pyomo
   ```

3. **Check for Issues**  
   If any errors occur, verify the package versions and dependencies using:
   ```bash
   conda list
   pip list
   ```

---

### **Additional Notes**

- If you encounter version conflicts, Conda will notify you. You can try resolving them by running:
  ```bash
  conda install --update-deps <package_name>
  ```
  
- For GPU support with PyTorch, ensure your system has compatible CUDA drivers installed. You can verify CUDA compatibility using:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```

---

### **Common Commands Reference**

| **Command**                             | **Description**                              |
|-----------------------------------------|----------------------------------------------|
| `conda activate <env_name>`             | Activate a Conda environment.               |
| `conda env update --file <file>.yaml`   | Update an environment with a YAML file.     |
| `conda install <package>`               | Install a package with Conda.               |
| `pip install <package>`                 | Install a package with pip.                 |
| `conda list`                            | List installed Conda packages.              |
| `pip list`                              | List installed pip packages.                |