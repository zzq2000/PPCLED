#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os

from data_convert.task_format.task_format import TaskFormat


class DyIEPP(TaskFormat):
    def __init__(self, doc_json):
        self.doc_key = doc_json['doc_key']
        self.sentences = doc_json['sentences']
        self.ner = doc_json['ner']
        self.relations = doc_json['relations']
        self.events = doc_json['events']
        self.sentence_start = doc_json.get('sentence_start', doc_json['_sentence_start'])

    def generate_sentence(self, type_format='subtype'):
        for sentence, events_in_sentence, sentence_start in zip(self.sentences, self.events, self.sentence_start):
            events = list()
            for event in events_in_sentence:
                trigger, event_type = event[0]
                trigger -= sentence_start

                suptype, subtype = event_type.split('.')

                if type_format == 'subtype':
                    event_type = subtype
                elif type_format == 'suptype':
                    event_type = suptype
                else:
                    event_type = suptype + type_format + subtype

                arguments = list()
                for start, end, role in event[1:]:
                    start -= sentence_start
                    end -= sentence_start
                    arguments += [[role, list(range(start, end + 1))]]

                event = {'type': event_type, 'tokens': [trigger], 'arguments': arguments}

                events += [event]
            yield {'tokens': sentence, 'events': events}


class Event(TaskFormat):
    """
    {
        "doc_id": "NYT_ENG_20130914.0094",
        "sent_id": "NYT_ENG_20130914.0094-1",
        "tokens": ["LARGO", "\u2014", "A", "judge", "on", "Friday", "refused", "to", "stop", "''", "Hiccup", "Girl", "\u2019'", "Jennifer", "Mee", "from", "giving", "interviews", "to", "the", "media", "in", "the", "final", "days", "before", "her", "murder", "trial", "."],
        "entities": [
            {"entity_id": "NYT_ENG_20130914.0094-1-8-42", "entity_type": "PER", "mention_type": "NOM", "start": 3, "end": 4, "text": "judge"},
            {"entity_id": "NYT_ENG_20130914.0094-1-39-2096", "entity_type": "PER", "mention_type": "NAM", "start": 10, "end": 12, "text": "Hiccup Girl"},
            {"entity_id": "NYT_ENG_20130914.0094-1-39-48", "entity_type": "PER", "mention_type": "NAM", "start": 13, "end": 15, "text": "Jennifer Mee"},
            {"entity_id": "NYT_ENG_20130914.0094-1-39-54", "entity_type": "PER", "mention_type": "PRO", "start": 26, "end": 27, "text": "her"}
        ],
        "relations": [],
        "events": [
            {
                "event_id": "NYT_ENG_20130914.0094-1-25-998", "event_type": "contact", "event_subtype": "broadcast",
                "trigger": {"text": "interviews", "start": 17, "end": 18},
                "arguments": [
                    {"entity_id": "NYT_ENG_20130914.0094-1-39-48", "role": "entity", "text": "Jennifer Mee"}
                ]
            },
            {
                "event_id": "NYT_ENG_20130914.0094-1-1040-1019", "event_type": "justice", "event_subtype": "trialhearing",
                "trigger": {"text": "trial", "start": 28, "end": 29},
                 "arguments": [{"entity_id": "NYT_ENG_20130914.0094-1-39-54", "role": "defendant", "text": "her"}
                 ]
            }
        ],
        "start": 231, "end": 380,
         "text": "LARGO \u2014 A judge on Friday refused to stop ''Hiccup Girl\u2019' Jennifer Mee from giving interviews to the media in the final days before her murder trial."
    }
    """

    def __init__(self, doc_json):
        self.doc_key = doc_json['doc_id']
        self.sentence = doc_json['tokens']
        self.entities = {entity['id']: entity for entity in doc_json['entity_mentions']}
        self.relations = doc_json['relation_mentions']
        self.events = doc_json['event_mentions']
        # self.sentence_start = doc_json['start']
        # self.sentence_end = doc_json['end']
        # self.text = doc_json['text']

    def generate_sentence(self, type_format='subtype'):
        events = list()

        for event in self.events:
            arguments = list()
            for argument in event['arguments']:
                argument_entity = self.entities[argument['entity_id']]
                arguments += [[argument['role'], list(range(argument_entity['start'], argument_entity['end']))]]

            suptype, subtype = event['event_type'].split(':')

            if type_format == 'subtype':
                event_type = subtype
            elif type_format == 'suptype':
                event_type = suptype
            else:
                event_type = suptype + type_format + subtype

            events += [{
                'type': event_type,
                'tokens': list(range(event['trigger']['start'], event['trigger']['end'])),
                'arguments': arguments
            }]

        yield {'tokens': self.sentence, 'events': events}


class CLED(TaskFormat):
    """
    CLED data_format:
    {
        "golden-entity-mentions": [
            {
                "end": 26,
                "entity-type": "PER:Individual",
                "id": "AFP_ENG_20030401.0476-E1",
                "phrase-type": "NAM",
                "start": 23,
                "text": "incumbent Howard Davies"
            },
            {
                "end": 24,
                "entity-type": "PER:Individual",
                "id": "AFP_ENG_20030401.0476-E1",
                "phrase-type": "NOMPRE",
                "start": 23,
                "text": "incumbent"
            },
            {
                "end": 5,
                "entity-type": "PER:Individual",
                "id": "AFP_ENG_20030401.0476-E5",
                "phrase-type": "NAM",
                "start": 0,
                "text": "Former senior banker Callum McCarthy"
            },
            {
                "end": 3,
                "entity-type": "PER:Individual",
                "id": "AFP_ENG_20030401.0476-E5",
                "phrase-type": "NOMPRE",
                "start": 0,
                "text": "Former senior banker"
            },
            {
                "end": 16,
                "entity-type": "GPE:Population-Center",
                "id": "AFP_ENG_20030401.0476-E14",
                "phrase-type": "NAM",
                "start": 15,
                "text": "London"
            },
            {
                "end": 21,
                "entity-type": "TIM:time",
                "id": "AFP_ENG_20030401.0476-T4",
                "phrase-type": "TIM",
                "start": 20,
                "text": "September"
            },
            {
                "end": 1,
                "entity-type": "TIM:time",
                "id": "AFP_ENG_20030401.0476-T8",
                "phrase-type": "TIM",
                "start": 0,
                "text": "Former"
            },
            {
                "end": 19,
                "entity-type": "Job-Title:other",
                "id": "AFP_ENG_20030401.0476-V1",
                "phrase-type": "VALUE",
                "start": 8,
                "text": "one of the most important jobs in London's financial world"
            }
        ],
        "golden-event-mentions": [
            {
                "arguments": [
                    {
                        "end": 5,
                        "role": "Person",
                        "start": 0,
                        "text": "Former senior banker Callum McCarthy"
                    },
                    {
                        "end": 19,
                        "role": "Position",
                        "start": 8,
                        "text": "one of the most\nimportant jobs in London's financial world"
                    },
                    {
                        "end": 21,
                        "role": "Time-Within",
                        "start": 20,
                        "text": "September"
                    }
                ],
                "event_type": "Personnel:Start-Position",
                "id": "AFP_ENG_20030401.0476-EV1",
                "trigger": {
                    "end": 6,
                    "start": 5,
                    "text": "begins"
                }
            },
            {
                "arguments": [
                    {
                        "end": 5,
                        "role": "Person",
                        "start": 0,
                        "text": "Former senior banker Callum McCarthy"
                    }
                ],
                "event_type": "Personnel:End-Position",
                "id": "AFP_ENG_20030401.0476-EV2",
                "trigger": {
                    "end": 1,
                    "start": 0,
                    "text": "Former"
                }
            },
            {
                "arguments": [
                    {
                        "end": 26,
                        "role": "Person",
                        "start": 23,
                        "text": "incumbent Howard Davies"
                    },
                    {
                        "end": 16,
                        "role": "Place",
                        "start": 15,
                        "text": "London"
                    },
                    {
                        "end": 19,
                        "role": "Position",
                        "start": 8,
                        "text": "one of the most\nimportant jobs in London's financial world"
                    },
                    {
                        "end": 21,
                        "role": "Time-Within",
                        "start": 20,
                        "text": "September"
                    }
                ],
                "event_type": "Personnel:End-Position",
                "id": "AFP_ENG_20030401.0476-EV3",
                "trigger": {
                    "end": 28,
                    "start": 26,
                    "text": "steps down"
                }
            }
        ],
        "penn_treebank": [],
        "pos-tags": [
            "JJ",
            "JJ",
            "NN",
            "NNP",
            "NNP",
            "VBZ",
            "WP",
            "VBZ",
            "CD",
            "IN",
            "DT",
            "RBS",
            "JJ",
            "NNS",
            "IN",
            "NNP",
            "POS",
            "JJ",
            "NN",
            "IN",
            "NNP",
            ",",
            "WRB",
            "JJ",
            "NNP",
            "NNP",
            "VBZ",
            "RP"
        ],
        "sentence": "Former senior banker Callum McCarthy begins what is one of the most important jobs in London's financial world in September, when incumbent Howard Davies steps down",
        "stanford-colcc": [],
        "words": [
            "Former",
            "senior",
            "banker",
            "Callum",
            "McCarthy",
            "begins",
            "what",
            "is",
            "one",
            "of",
            "the",
            "most",
            "important",
            "jobs",
            "in",
            "London",
            "'s",
            "financial",
            "world",
            "in",
            "September",
            ",",
            "when",
            "incumbent",
            "Howard",
            "Davies",
            "steps",
            "down"
        ]
    }
    """
    def __init__(self, doc_json):
        self.entity_mentions = doc_json["golden-entity-mentions"]
        self.event_mentions = doc_json["golden-event-mentions"]
        self.sentence = doc_json["sentence"]

    def generate_sentence(self, type_format='subtype'):
        events = list()

        for event in self.event_mentions:
            arguments = list()
            for argument in event['arguments']:
                arguments += [[argument['role'], argument['text']]]

            suptype, subtype = event['event_type'].split(':')

            if type_format == 'subtype':
                event_type = subtype
            elif type_format == 'suptype':
                event_type = suptype
            else:
                event_type = suptype + type_format + subtype

            events += [{
                'type': event_type,
                'tokens': event['trigger']['text'],
                'arguments': arguments
            }]

        yield {'tokens': self.sentence, 'events': events}


def DyIEPP_ace2005_file_tuple(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    conll_2012_folder = "data/raw_data/dyiepp_ace2005"

    file_tuple = [
        (conll_2012_folder + "/train.json", output_folder + '/train'),
        (conll_2012_folder + "/dev.json", output_folder + '/val'),
        (conll_2012_folder + "/test.json", output_folder + '/test'),
    ]

    return file_tuple


def ace2005_en_file_tuple(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    conll_2012_folder = "data/raw_data/ace05-EN/"

    file_tuple = [
        (conll_2012_folder + "/train.oneie.json", output_folder + '/train'),
        (conll_2012_folder + "/dev.oneie.json", output_folder + '/val'),
        (conll_2012_folder + "/test.oneie.json", output_folder + '/test'),
    ]

    return file_tuple


def ere_en_file_tuple(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    conll_2012_folder = "data/raw_data/ERE-EN"

    file_tuple = [
        (conll_2012_folder + "/train.oneie.json", output_folder + '/train'),
        (conll_2012_folder + "/dev.oneie.json", output_folder + '/val'),
        (conll_2012_folder + "/test.oneie.json", output_folder + '/test'),
    ]

    return file_tuple


def cled_file_tuple(output_folder, language):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    conll_2012_folder = "data/raw_data/CLED" + language
    output_folder = output_folder + language

    file_tuple = [
        (conll_2012_folder + "/train_with_neg_eg.json", output_folder + '/train'),
        (conll_2012_folder + "/dev_with_neg_eg.json", output_folder + '/val'),
        (conll_2012_folder + "/test_with_neg_eg.json", output_folder + '/test'),
    ]

    return file_tuple


if __name__ == "__main__":
    pass
