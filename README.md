# AlpaGasus
The unofficial implementation of "AlpaGasus: Training a better Alpaca with Fewer data."

## Setup
- Set up the environment of [Alpaca](https://github.com/tatsu-lab/stanford_alpaca).

## Rating
Rate each (instruction, input, output) tuple in the Alpaca's 52k training set.
```
# Use ChatGPT as the response quality evaluator
export YOUR_OPENAI_API_KEY
# Use Claude as the response quality evaluator
export YOUR_CLAUDE_API_KEY
```
After the rating, you will need to use `rating/filter.py` and `rating/get_scores.py` to process your reviews obtained from ChatGPT/Claude.
The final results here are in `claude_t45.json`.


## Training
- For the instruction-finetuning of LLaMA-7B: 
```
# prepare the data 
sh training/train_7b.sh
```
- For the instruction-finetuning of LLaMA-13B:
```
sh training/train_13b.sh
```



## Other tests
We use ChatGPT as the grader to evaluate the model's output.
```
export OPENAI_API_KEY
cd evaluation/
sh run_eval.sh
```


## References
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [Koala](https://github.com/young-geng/EasyLM/tree/main)
- [Vicuna](https://vicuna.lmsys.org/)
- [GPT-4-Report](https://arxiv.org/pdf/2303.08774.pdf)

## Citation
If you think it is a useful repo, please cite the paper:
```
@article{chen2023instructzero,
  title={InstructZero: Efficient Instruction Optimization for Black-Box Large Language Models},
  author={Chen, Lichang and Chen, Jiuhai and Goldstein, Tom and Huang, Heng and Zhou, Tianyi},
  journal={arXiv preprint arXiv:2306.03082},
  year={2023}
}
```