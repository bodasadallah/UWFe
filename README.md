Repo for useful Writing Feedback project.

## Main repo link: https://github.com/bodasadallah/UWFe

### Run Fine-tuning
* The main fine-tuning code is available in the `finetune.py` file.
* We use all available `HFTrainingArguments`. in addition to custom variables that are available at the `arguemnts.py` module.
* You can see some of the default parameters at `tain.sh` script

### Run evaluation
```bash
bash or sbatch eval_boda.sh
```
