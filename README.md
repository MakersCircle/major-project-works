# **Accident Anticipation Project - Setup and Guidelines**

## **Project Setup**

### **1. Clone the Repository**
Start by cloning this repository and setting up the Python environment.

### **2. Create a Conda Environment**
Ensure you have Conda installed. Create a new environment with Python 3.10:
```bash
conda create -n accident-anticipation python=3.10
```
If you're using PyCharm, select this environment as the project's interpreter.

### **3. Install PyTorch and Dependencies**
For systems with CUDA, verify your CUDA version and find the corresponding PyTorch installation command on the [PyTorch Get Started page](https://pytorch.org/get-started/locally/). 

For example, with CUDA 12.4, use:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```
Check your system compatability before installation

---

## **Guidelines**

### **Avoid Pushing Large Binary Files**
Do not push the following to the repository:
- Dataset images
- Saved model files
- Other large binary files

To prevent accidental uploads, ensure these files are listed in the [`.gitignore`](.gitignore) file.

### **Begin writing code in the [src](src/) directory.**