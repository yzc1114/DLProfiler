from transformers import AutoModelForSequenceClassification, BertConfig


def load_bert_base():
    return AutoModelForSequenceClassification.from_config(BertConfig())


def do_test():
    model1 = load_bert_base()
    print(model1)


if __name__ == '__main__':
    do_test()
