STEP 01- Download Llama3.1 locally from Ollama in Ubuntu

    sudo ufw allow 11434/tcp
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.1:8b
    ollama run llama3.1

STEP 02- Create a conda environment after opening the repository

sudo apt install python3-virtualenv
virtualenv ragpipeline
source ragpipeline/bin/activate
deactivate

STEP 03- install the requirements

pip install -r requirements.txt