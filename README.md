---
license: apache-2.0
---

# Burmese RoBERTa
You can access the model from the [HuggingFace Repository](https://huggingface.co/saihtaungkham/BurmeseRoBERTa).

## Description
The model is adopted from the RoBERTa base model and trained using Masked Language Modeling (MLM) with the following datasets:

1. `oscar-corpus/OSCAR-2301`
2. `5w4n/OSCAR-2019-Burmese-fix`
3. Wikipedia
4. [myParaphrase](https://github.com/ye-kyaw-thu/myParaphrase)
5. [myanmar_news](https://huggingface.co/datasets/myanmar_news)
6. [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200)
7. [myPOS](https://github.com/ye-kyaw-thu/myPOS.git)
8. [BurmeseProverbDataSet](https://github.com/vincent-paing/BurmeseProverbDataSet.git)
9. [TALPCo](https://github.com/matbahasa/TALPCo.git)

## Model Usage


```python
from transformers import pipeline

model_name = "saihtaungkham/BurmeseRoBERTa"

fill_mask = pipeline(
    "fill-mask",
    model=model_name,
    tokenizer=model_name,
)

print(fill_mask("ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ [MASK] ဖြစ်သည်။"))
```

```shell
[{'score': 0.5182967782020569,
  'token': 1071,
  'token_str': 'မြို့တော်',
  'sequence': 'ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ မြို့တော် ဖြစ်သည်။'},
 {'score': 0.029216164723038673,
  'token': 28612,
  'token_str': 'အကြီးဆုံးမြို့',
  'sequence': 'ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ အကြီးဆုံးမြို့ ဖြစ်သည်။'},
 {'score': 0.013689162209630013,
  'token': 2034,
  'token_str': 'လေဆိပ်',
  'sequence': 'ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ လေဆိပ် ဖြစ်သည်။'},
 {'score': 0.01367204450070858,
  'token': 17641,
  'token_str': 'ရုံးစိုက်ရာမြို့',
  'sequence': 'ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ ရုံးစိုက်ရာမြို့ ဖြစ်သည်။'},
 {'score': 0.010110817849636078,
  'token': 2723,
  'token_str': 'အရှေ့ပိုင်း',
  'sequence': 'ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ အရှေ့ပိုင်း ဖြစ်သည်။'}]
```

## How to use only the trained tokenizer for Burmese sentences
```python
from transformers import AutoTokenizer

model_name = "saihtaungkham/BurmeseRoBERTa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
text = "သဘာဝဟာသဘာဝပါ။"

# Tokenized words
print(tokenizer.tokenize(text))
# Expected Output
# ['▁', 'သဘာဝ', 'ဟာ', 'သဘာဝ', 'ပါ။']

# Tokenized IDs for training other models
print(tokenizer.encode(text))
# Expected Output
# [1, 3, 1003, 30, 1003, 62, 2]

```

## Extract text embedding from the sentence
```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "saihtaungkham/BurmeseRoBERTa"

# Loading the model
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Sample data
input_texts = [
    "ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ စီးပွားရေးမြို့တော်ဖြစ်သည်။",
    "ဘန်ကောက်သည် ထိုင်းနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။",
    "နေပြည်တော်သည် မြန်မာနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။",
    "ဂျပန်ကို အလည်သွားမယ်။",
    "ဗိုက်ဆာတယ်။",
    "ထိုင်းအစားအစာကို ကြိုက်တယ်။",
    "ခွေးလေးကချစ်စရာလေး",
    "မင်းသမီးလေးက ချစ်စရာလေး"
]

# Function for encode our sentences
def encode(inputs):
    return tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt",
    )


# Enter the evaluation mode
model.eval()

for idx in range(len(input_texts)):
    target_sentence = input_texts[idx]
    compare_sentences = input_texts[:]
    compare_sentences.remove(target_sentence)
    outputs = []
    with torch.no_grad():
        for token in compare_sentences:
            model_output = model(**encode([target_sentence, token]))
            # If you would like to extract the sentences embedding,
            # the following line does the job for you.
            sentence_embeddings = model_output[0].mean(dim=1)

            # Check the sentence similarity.
            similarity_score = torch.nn.functional.cosine_similarity(
                sentence_embeddings[0].reshape(1, -1), 
                sentence_embeddings[1].reshape(1, -1)
            )
            outputs.append((target_sentence, token, similarity_score.item()))
            # print(f"{target_sentence} vs {token} => {similarity_score}")

    print("*" * 50)
    # Sort the score in descending order.
    outputs.sort(key=lambda x: x[2], reverse=True)
    top_k = 3
    [print(result) for result in outputs[:top_k]]
```

```shell
**************************************************
('ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ စီးပွားရေးမြို့တော်ဖြစ်သည်။', 'နေပြည်တော်သည် မြန်မာနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 0.9941556453704834)
('ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ စီးပွားရေးမြို့တော်ဖြစ်သည်။', 'ဘန်ကောက်သည် ထိုင်းနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 0.9840704202651978)
('ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ စီးပွားရေးမြို့တော်ဖြစ်သည်။', 'ဂျပန်ကို အလည်သွားမယ်။', 0.9625985026359558)
**************************************************
('ဘန်ကောက်သည် ထိုင်းနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 'ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ စီးပွားရေးမြို့တော်ဖြစ်သည်။', 0.9840705394744873)
('ဘန်ကောက်သည် ထိုင်းနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 'နေပြည်တော်သည် မြန်မာနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 0.9832078814506531)
('ဘန်ကောက်သည် ထိုင်းနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 'ဂျပန်ကို အလည်သွားမယ်။', 0.9640133380889893)
**************************************************
('နေပြည်တော်သည် မြန်မာနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 'ရန်ကုန်သည် မြန်မာနိုင်ငံ၏ စီးပွားရေးမြို့တော်ဖြစ်သည်။', 0.9941557049751282)
('နေပြည်တော်သည် မြန်မာနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 'ဘန်ကောက်သည် ထိုင်းနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 0.9832078218460083)
('နေပြည်တော်သည် မြန်မာနိုင်ငံ၏ မြို့တော်ဖြစ်သည်။', 'ဂျပန်ကို အလည်သွားမယ်။', 0.9571995139122009)
**************************************************
('ဂျပန်ကို အလည်သွားမယ်။', 'ဗိုက်ဆာတယ်။', 0.9784848093986511)
('ဂျပန်ကို အလည်သွားမယ်။', 'ထိုင်းအစားအစာကို ကြိုက်တယ်။', 0.9755436182022095)
('ဂျပန်ကို အလည်သွားမယ်။', 'မင်းသမီးလေးက ချစ်စရာလေး', 0.9682475924491882)
**************************************************
('ဗိုက်ဆာတယ်။', 'ဂျပန်ကို အလည်သွားမယ်။', 0.9784849882125854)
('ဗိုက်ဆာတယ်။', 'ထိုင်းအစားအစာကို ကြိုက်တယ်။', 0.9781478047370911)
('ဗိုက်ဆာတယ်။', 'ခွေးလေးကချစ်စရာလေး', 0.971768856048584)
**************************************************
('ထိုင်းအစားအစာကို ကြိုက်တယ်။', 'ဗိုက်ဆာတယ်။', 0.9781478047370911)
('ထိုင်းအစားအစာကို ကြိုက်တယ်။', 'ဂျပန်ကို အလည်သွားမယ်။', 0.975543737411499)
('ထိုင်းအစားအစာကို ကြိုက်တယ်။', 'မင်းသမီးလေးက ချစ်စရာလေး', 0.9729770421981812)
**************************************************
('ခွေးလေးကချစ်စရာလေး', 'မင်းသမီးလေးက ချစ်စရာလေး', 0.996442437171936)
('ခွေးလေးကချစ်စရာလေး', 'ဗိုက်ဆာတယ်။', 0.971768856048584)
('ခွေးလေးကချစ်စရာလေး', 'ထိုင်းအစားအစာကို ကြိုက်တယ်။', 0.9697750806808472)
**************************************************
('မင်းသမီးလေးက ချစ်စရာလေး', 'ခွေးလေးကချစ်စရာလေး', 0.9964425563812256)
('မင်းသမီးလေးက ချစ်စရာလေး', 'ထိုင်းအစားအစာကို ကြိုက်တယ်။', 0.9729769229888916)
('မင်းသမီးလေးက ချစ်စရာလေး', 'ဗိုက်ဆာတယ်။', 0.9686307907104492)
```

# Credit
I thank the original author and contributor mentioned in the dataset sections.
We have the technologies but need the datasets to make the model work. The transformer model has been available since 2017. However, it is still challenging to train the model due to the low language resources available over the internet. This model will be a stepping stone for us to create a more open model for the Myanmar language and benefit our community.
Anyone is welcome to contact me regarding the license and contribution to the improvement of this model.
