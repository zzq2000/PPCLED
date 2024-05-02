#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
import json
import collections

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, input, output):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            input: The input sequence.
            output: ground truth output.
        """
        self.guid = guid
        self.input = input
        self.output = output

Prefix = "Classify event type:"
PROMPT = "The event type of {} is"

def read_examples_from_file(data_dir, mode, dataset):
    if dataset == "ACE":
        file_path = os.path.join(data_dir, "{}_wout_neg_eg.json".format(mode))
    else:
        file_path = os.path.join(data_dir, "{}.json".format(mode))
    examples = []
    words=[]
    labels=[]
    
    def getLabel(eT,is_start=False):
        label = ''
        if eT=='None':
            label = 'O'
        elif is_start:
            label = "B-"+eT
        else: label = "I-"+eT
        if label not in set(get_labels(dataset)):
            print('error event type: ' + eT)
            exit(1)
        return label

    if dataset == "ACE":
        datas=json.load(open(file_path, "r", encoding="utf-8"))
        for i, data in enumerate(datas):
            words=data['words']
            labels=['O' for i in range(0,len(words))]
            if len(data['golden-event-mentions']) != 0:
                for event in data['golden-event-mentions']:
                    trigger_start = event['trigger']['start']
                    trigger_end = event['trigger']['end']
                    trigger_text = event['trigger']['text']
                    event_type = event['event_type']

                    if trigger_start == -1:
                        sentence = data['sentence']
                        words_offset = [(sentence.find(word), sentence.find(word) + len(word) - 1) for word in words]
                        trigger_str_start = sentence.find(trigger_text)
                        trigger_str_end = sentence.find(trigger_text) + len(trigger_text) - 1
                        for j, (start, end) in enumerate(words_offset):
                            if trigger_str_start >= start and trigger_str_start < end:
                                trigger_start = j
                            if trigger_str_end >= start and trigger_str_end <= end:
                                trigger_end = j
                            if trigger_start != -1 and trigger_end != -1:
                                break

                    labels[trigger_start] = getLabel((event_type.split(":"))[1], True)
                    for k in range(trigger_start+1, trigger_end):
                        labels[k]=getLabel((event_type.split(":"))[1])
                
                for (word, label) in zip(words, labels):
                    input = Prefix + " " + data["sentence"] + " " + PROMPT.format(word)
                    output = label

                    examples.append({"input": input, "output": output})
    else:
        with open(file_path, "r", encoding="utf-8") as fin:
            for line in fin.readlines():
                data = json.loads(line)
                words=data['tokens']
                labels=data["labels"]
                for (word, label) in zip(words, labels):
                        input = Prefix + " " + data["sentence"] + " " + PROMPT.format(word)
                        output = label
                        examples.append({"input": input, "output": output})
    
    return examples


def get_labels(dataset):
    if dataset == "ACE":
        return ["O", "B-Attack", "I-Attack", "B-Transport", "I-Transport", "B-Die", "I-Die", "B-End-Position", "I-End-Position", "B-Meet", "I-Meet", "B-Phone-Write", "I-Phone-Write", "B-Elect", "I-Elect", "B-Injure", "I-Injure", "B-Transfer-Ownership", "I-Transfer-Ownership", "B-Start-Org", "I-Start-Org", "B-Transfer-Money", "I-Transfer-Money", "B-Sue", "I-Sue", "B-Demonstrate", "I-Demonstrate", "B-Arrest-Jail", "I-Arrest-Jail", "B-Start-Position", "I-Start-Position", "B-Be-Born", "I-Be-Born", "B-End-Org", "I-End-Org", "B-Execute", "I-Execute", "B-Nominate", "I-Nominate", "B-Fine", "I-Fine", "B-Trial-Hearing", "I-Trial-Hearing", "B-Marry", "I-Marry", "B-Charge-Indict", "I-Charge-Indict", "B-Sentence", "I-Sentence", "B-Convict", "I-Convict", "B-Appeal", "I-Appeal", "B-Declare-Bankruptcy", "I-Declare-Bankruptcy", "B-Merge-Org", "I-Merge-Org", "B-Release-Parole", "I-Release-Parole", "B-Pardon", "I-Pardon", "B-Extradite", "I-Extradite", "B-Divorce", "I-Divorce", "B-Acquit", "I-Acquit"]
    else:
        return ["O", "B-Be-Born", "I-Be-Born", "B-Marry", "I-Marry", "B-Divorce", "I-Divorce", "B-Injure", "I-Injure", "B-Die", "I-Die", "B-Transport", "I-Transport", "B-Transfer-Ownership", "I-Transfer-Ownership", "B-Transfer-Money", "I-Transfer-Money", "B-Attack", "I-Attack", "B-Demonstrate", "I-Demonstrate", "B-Meet", "I-Meet", "B-Phone-Write", "I-Phone-Write", "B-Start-Position", "I-Start-Position", "B-End-Position", "I-End-Position", "B-Arrest-Jail", "I-Arrest-Jail", "B-START-ORG", "I-START-ORG"]


if __name__ == "__main__":
    base_dir = "./data"
    for dataset in ["ACE", "MINION"]:
        if dataset == "ACE":
            langs = ["English", "Chinese", "Arabic"]
        else:
            langs = ["en", "es", "hi", "ja", "ko", "pl", "pt", "tr"]
        for lang in langs:
            for mode in ["train", "dev", "test"]:
                examples = read_examples_from_file(os.path.join(base_dir, dataset, lang), mode, dataset)
                fout_path = os.path.join(base_dir, dataset, lang, mode + "_t5.json")
                with open(fout_path, "w", encoding="utf-8") as fout:
                    json.dump(examples, fout, ensure_ascii=False, indent=4)
            
            
