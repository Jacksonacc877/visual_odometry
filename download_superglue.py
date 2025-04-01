#!/usr/bin/env python
"""
Script to download and set up SuperGlue for the visual odometry project.
"""
import os
import sys
import subprocess
import argparse

def main():
    print("Setting up SuperGlue for visual odometry...")
    
    # Define the target directory
    target_dir = os.path.join(os.getcwd(), 'SuperGluePretrainedNetwork')
    
    # Check if the directory already exists
    if os.path.exists(target_dir):
        print(f"SuperGlue directory already exists at {target_dir}")
        print("Updating repository...")
        try:
            os.chdir(target_dir)
            subprocess.check_call(['git', 'pull'])
            os.chdir('..')
            print("Successfully updated SuperGlue")
        except subprocess.CalledProcessError:
            print("Failed to update SuperGlue. It may not be a git repository.")
        except Exception as e:
            print(f"An error occurred while updating: {e}")
    else:
        # Clone the repository
        print(f"Cloning SuperGlue repository to {target_dir}...")
        try:
            subprocess.check_call([
                'git', 'clone', 
                'https://github.com/magicleap/SuperGluePretrainedNetwork.git',
                target_dir
            ])
            print("Successfully cloned SuperGlue repository")
        except subprocess.CalledProcessError:
            print("Failed to clone SuperGlue repository. Please check your internet connection and git installation.")
            return
        except Exception as e:
            print(f"An error occurred while cloning: {e}")
            return
    
    # Install requirements
    print("Installing required packages for SuperGlue...")
    requirements_file = os.path.join(target_dir, 'requirements.txt')
    
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-r', requirements_file
            ])
            print("Successfully installed SuperGlue requirements")
        except subprocess.CalledProcessError:
            print("Failed to install requirements. You may need to install them manually.")
        except Exception as e:
            print(f"An error occurred while installing requirements: {e}")
    else:
        print(f"Requirements file not found at {requirements_file}")
        print("You may need to install dependencies manually.")
    
    # Install PyTorch if needed
    try:
        import torch
        print(f"PyTorch {torch.__version__} is already installed")
    except ImportError:
        print("PyTorch not found. Installing PyTorch...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'
            ])
            print("Successfully installed PyTorch")
        except subprocess.CalledProcessError:
            print("Failed to install PyTorch. Please install it manually from https://pytorch.org/")
        except Exception as e:
            print(f"An error occurred while installing PyTorch: {e}")
    
    print("\nSetup complete!")
    print("To use SuperGlue in your visual odometry code, run:")
    print("  python vo.py <your_input_directory>")
    print("\nNote: SuperGlue requires significant computational resources.")
    print("If you encounter performance issues, you may want to reduce the number of keypoints")
    print("or use a smaller subset of your dataset.")

if __name__ == "__main__":
    main()
