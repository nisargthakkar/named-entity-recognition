import os

def preprocess(dataFileName='./eng.train', outfolder='train_data/'):
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
      elif line == '-DOCSTART- -X- -X- O\n':
        pass
      else:
        lines += line
  
  f = open(outfolder + '/' + metafilename, 'w')
  f.write('%s\n%s\n'%(next_file_index, fileprefix))
  f.close()
  pass


if __name__ == '__main__':
  preprocess(dataFileName = './eng.train', outfolder ='train_data/')
  preprocess(dataFileName = './eng.testa', outfolder ='testa_data/')
  preprocess(dataFileName = './eng.testb', outfolder ='testb_data/')