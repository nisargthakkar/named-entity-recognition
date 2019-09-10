import os

import os

def preprocess(dataFileName='./AIDA-YAGO2-dataset.tsv', outfolder='aida_yago_wikipedia/'):
  if os.path.exists(outfolder):
    print('%s exists. Skipping it' % outfolder)
    return
  os.mkdir(outfolder)
  metafilename='__META__'
  fileprefix='sentence_'
  lines = ''
  next_file_index = 0
  with open(dataFileName, 'r') as f:
    for line in f:
      if len(line) == 1: # Only \n
        if len(lines) == 0:
          continue
        outfile = open(outfolder + '/' + fileprefix + str(next_file_index), 'w')
        outfile.write(lines)
        outfile.close()
        next_file_index += 1
        lines = ''
      elif '-DOCSTART-' in line:
        pass
      else:
        lines += line
  
  f = open(outfolder + '/' + metafilename, 'w')
  f.write('%s\n%s\n'%(next_file_index, fileprefix))
  f.close()
  pass


# def preprocess(dataFileName='./AIDA-YAGO2-dataset.tsv', outfile='aida_yago_wikipedia.txt'):
#     if os.path.exists(outfile):
#         print('%s exists. Skipping it' % outfolder)
#         return
#     entity_article_dict = {}
#     with open(dataFileName, 'r') as f:
#         previous_entity = ''
#         for line in f:
#             if 'http://en.wikipedia.org' in line:
#                 columns = line.split('\t')
#                 complete_entity = columns[2]
#                 wikipedia_article = columns[4]
#                 if complete_entity not in entity_article_dict:
#                     entity_article_dict[complete_entity] = []
#                 entity_article_dict[complete_entity].append(wikipedia_article)

#     f = open(outfile, 'w')
#     lines = '\n'.join(map(lambda key: key + '\t' + '\t'.join(entity_article_dict[key]), entity_article_dict.keys()))
#     f.write(lines)
#     f.close()
#     pass


if __name__ == '__main__':
  preprocess(dataFileName='./AIDA-YAGO2-dataset.tsv', outfolder='aida_yago_wikipedia/')