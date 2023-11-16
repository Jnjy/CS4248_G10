# CS4248 Group Project: Machine Reading Comprehension (MRC)

## Introduction
Machine Reading Comprehension (MRC) is a critical facet of Natural Language Processing (NLP), particularly in the domain of Question Answering (QA) tasks. Question Answering involves comprehending the given corpus, then selecting the span of text in the corpus that answers to a given set of questions. In this project, we explored the different pre-trained transformer-based models to identify the strengths and weaknesses of the different models, respectively BERT, DistilBERT, RoBERTa, AlBERT and XLNet. To further improve the performance on QA, we proceed to fine-tune the models' hyperparameters to obtain best performing based model. We then explore the imperative of enhancing MRC performance through an ensemble methodology, amalgamating the capabilities of the fine-tuned transformer-based models. Our experiments demonstrates that **our ensemble model outperforms** all of the selected baseline models.

## Built With
The project is built using the following technologies and frameworks:
- [Python](https://www.python.org/)
- [Hugging Face](https://huggingface.co/)
- [BERT](https://github.com/google-research/bert)
- [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- [ALBERT](https://huggingface.co/albert-base-v2)
- [RoBERTa](https://huggingface.co/roberta-base)
- [XLNet](https://huggingface.co/xlnet-base-cased)
- [Jupyter Notebook](https://jupyter.org/)

## Team Members of Group 10
- A0170723L
- A0241293J
- A0240932L
- A0238397M
- A0236491B


## All finetuned models are uploaded onto HuggingFace
- jeffnjy/bert-base-test
- jeffnjy/albert-base-test
- jeffnjy/distilbert-base-test
- jeffnjy/roberta-base-test
- JiayanL/XLNET

## Getting Started
To run the project locally, follow these steps:

1. Clone the repository to your local machine:
   ```bash
   git clone git@github.com:Jnjy/CS4248_G10.git
   ```

2. Navigate to the project directory:
   ```bash
   cd CS4248_G10
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. To run the ensemble method:
   
   #1: Open `src/ensemble.py`

   #2: Select the ensemble method: hardmax / softmax by uncommenting the method you wish to run and comment the other
   ```
   if __name__ == '__main__':
      run_soft_ensemble()
      # run_hard_ensemble()
   ```

   #3: Open `script/run_ensemble.sh` to change your conda configuration
   ```
   ## edit the file path to your conda file env filepath
   source /home/n/njinyuan/miniconda3/etc/profile.d/conda.sh

   ## edit the environment to your conda environment
   echo "activating environment"
   conda activate cs4248
   ```

   #4: In the root folder, run
   ```bash
   sbatch script/run_ensemble.sh
   ```

5. Open the slurmog to check for the evaluated results.

Now, you're ready to explore and experiment with our MRC project. Feel free to customize configurations and parameters based on your requirements in `src/ensemble.py`.
```python
# line 203
''' 2. create ensemble model '''
soft_weights = [(1, 1, 1, 1)]

# line 270
''' 2. create ensemble model '''
soft_weights = [(1, 1, 1, 1)]
```

Data on our experiments can be found here: https://docs.google.com/document/d/1DR2qqxTZ056ni9E6luCJ8e4A7bjkjCG0uhZH-JnLzRc/edit?usp=sharing 

Happy coding!
