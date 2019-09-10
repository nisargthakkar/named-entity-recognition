from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from enum import Enum

from conll_util import loadDataFromFile

# def get_labels():
#   return ["B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]

class Tag_Types(Enum):
  ORGANIZATION = 1
  PERSON = 2
  LOCATION = 3
  DATE = 4
  TIME = 5
  MONEY = 6
  PERCENT = 7
  FACILITY = 8
  GPE = 9
  O = 11
  @staticmethod
  def from_str(label):
    if 'ORGANIZATION' in label:
      return Tag_Types.ORGANIZATION
    if 'PERSON' in label:
      return Tag_Types.PERSON
    if 'LOCATION' in label:
      return Tag_Types.LOCATION
    if 'DATE' in label:
      return Tag_Types.DATE
    if 'TIME' in label:
      return Tag_Types.TIME
    if 'MONEY' in label:
      return Tag_Types.MONEY
    if 'PERCENT' in label:
      return Tag_Types.PERCENT
    if 'FACILITY' in label:
      return Tag_Types.FACILITY
    if 'GPE' in label:
      return Tag_Types.LOCATION
    else:
      return Tag_Types.O

# New only starts with B- if previous was of the same type. Else, start with I-
class Conll_Tags(Enum):
  I_ORG = 1
  B_ORG = 2
  I_PER = 3
  B_PER = 4
  I_LOC = 5
  B_LOC = 6
  I_MISC = 7
  B_MISC = 8
  O = 9
  @staticmethod
  def toString(tag):
    return {
      Conll_Tags.I_ORG: 'I-ORG',
      Conll_Tags.B_ORG: 'B-ORG',
      Conll_Tags.I_PER: 'I-PER',
      Conll_Tags.B_PER: 'B-PER',
      Conll_Tags.I_LOC: 'I-LOC',
      Conll_Tags.B_LOC: 'B-LOC',
      Conll_Tags.I_MISC: 'I-MISC',
      Conll_Tags.B_MISC: 'B-MISC',
      Conll_Tags.O: 'O',
    }[tag]

# New always starts with B-. I- is only for continuous
# class NLTK_Tags(Enum):
#   I_ORGANIZATION = 1
#   B_ORGANIZATION = 2
#   I_PERSON = 3
#   B_PERSON = 4
#   I_LOCATION = 5
#   B_LOCATION = 6
#   I_DATE = 7
#   B_DATE = 8
#   I_TIME = 9
#   B_TIME = 10
#   I_MONEY = 11
#   B_MONEY = 12
#   I_PERCENT = 13
#   B_PERCENT = 14
#   I_FACILITY = 15
#   B_FACILITY = 16
#   I_GPE = 17
#   B_GPE = 18
#   O = 19

# NLTK_TAGS_TO_TAG_TYPE = {
#   NLTK_Tags.I_ORGANIZATION : Tag_Types.ORGANIZATION,
#   NLTK_Tags.B_ORGANIZATION : Tag_Types.ORGANIZATION,
#   NLTK_Tags.I_PERSON : Tag_Types.PERSON,
#   NLTK_Tags.B_PERSON : Tag_Types.PERSON,
#   NLTK_Tags.I_LOCATION : Tag_Types.LOCATION,
#   NLTK_Tags.B_LOCATION : Tag_Types.LOCATION,
#   NLTK_Tags.I_DATE : Tag_Types.DATE,
#   NLTK_Tags.B_DATE : Tag_Types.DATE,
#   NLTK_Tags.I_TIME : Tag_Types.TIME,
#   NLTK_Tags.B_TIME : Tag_Types.TIME,
#   NLTK_Tags.I_MONEY : Tag_Types.MONEY,
#   NLTK_Tags.B_MONEY : Tag_Types.MONEY,
#   NLTK_Tags.I_PERCENT : Tag_Types.PERCENT,
#   NLTK_Tags.B_PERCENT : Tag_Types.PERCENT,
#   NLTK_Tags.I_FACILITY : Tag_Types.FACILITY,
#   NLTK_Tags.B_FACILITY : Tag_Types.FACILITY,
#   NLTK_Tags.I_GPE : Tag_Types.GPE,
#   NLTK_Tags.B_GPE : Tag_Types.GPE,
#   NLTK_Tags.O : Tag_Types.O,
# }

def get_organization_tag(previous):
  if previous == Tag_Types.ORGANIZATION:
    return Conll_Tags.B_ORG
  else:
    return Conll_Tags.I_ORG

def get_person_tag(previous):
  if previous == Tag_Types.PERSON:
    return Conll_Tags.B_PER
  else:
    return Conll_Tags.I_PER

def get_location_tag(previous):
  if previous == Tag_Types.LOCATION:
    return Conll_Tags.B_LOC
  else:
    return Conll_Tags.I_LOC

def get_misc_tag(nltk_tag, previous):
  if previous == nltk_tag:
    return Conll_Tags.B_MISC
  else:
    return Conll_Tags.I_MISC

# PER, LOC, ORG, MISC
def nltk_to_conll2003_tags(nltk_tags):
  previous = None
  conll_tags = []
  for nltk_tag in nltk_tags:
    newTag = Conll_Tags.O
    tagType = Tag_Types.from_str(nltk_tag)
    if nltk_tag == 'I-ORGANIZATION':
      newTag = Conll_Tags.I_ORG
    elif nltk_tag == 'B-ORGANIZATION':
      newTag = get_organization_tag(previous)
    elif nltk_tag == 'I-PERSON':
      newTag = Conll_Tags.I_PER
    elif nltk_tag == 'B-PERSON':
      newTag = get_person_tag(previous)
    # elif nltk_tag == 'I-LOCATION':
    #   newTag = Conll_Tags.I_LOC
    elif nltk_tag in ['I-LOCATION', 'B-LOCATION', 'I-GPE', 'B-GPE']:
      newTag = get_location_tag(previous)
    elif nltk_tag == 'O':
      newTag = Conll_Tags.O
    else:
      newTag = get_misc_tag(tagType, previous)

    previous = tagType
    conll_tags.append(newTag)

  return conll_tags

def report_results(filename, sentences, true_tags, predicted_tags, predicted_nltk_tags):
  f = open(filename, "w")
  for sentenceIndex in range(len(true_tags)):
    for tagIndex in range(len(true_tags[sentenceIndex])):
      f.write('%s\t%s\t%s\t%s\n' % (sentences[sentenceIndex][tagIndex][0], predicted_nltk_tags[sentenceIndex][tagIndex], true_tags[sentenceIndex][tagIndex], predicted_tags[sentenceIndex][tagIndex]))
    f.write('\n')
  f.close()

def perform_ner(dataFileName = './eng.train', outfile ='./output_file'):
  sentences, tags = loadDataFromFile(dataFileName)

  predicted_tags = []
  predicted_nltk_tags = []

  for sentence_data in sentences:
    tokens = list(map(lambda token_data: token_data[0], sentence_data))
    pos_tags = pos_tag(tokens)
    ne_chunks = ne_chunk(pos_tags)
    iob_tagged = tree2conlltags(ne_chunks)
    nltk_tags = list(map(lambda tag_data: tag_data[2], iob_tagged))
    conlltags = list(map(Conll_Tags.toString, nltk_to_conll2003_tags(nltk_tags)))
    predicted_tags.append(conlltags)
    predicted_nltk_tags.append(nltk_tags)

  report_results(outfile, sentences, tags, predicted_tags, predicted_nltk_tags)

if __name__ == '__main__':
  perform_ner(dataFileName = './eng.train', outfile ='./output_file.train')
  perform_ner(dataFileName = './eng.testa', outfile ='./output_file.testa')
  perform_ner(dataFileName = './eng.testb', outfile ='./output_file.testb')


"""
IOB2 format: Sentences are separated by empty lines. Each line contains
at least three columns, separated by whitespaces. The second last column
is the chunk tag according to the corpus, and the last column is the
predicted chunk tag.
"""