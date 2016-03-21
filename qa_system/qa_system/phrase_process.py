import re 

def parse(s):
  itr = iter(filter(lambda x: x, re.split("\\s+", s.replace('(', ' ( ').replace(')', ' ) '))))

  def _parse():
    stuff = []
    for x in itr:
      if x == ')':
        return stuff
      elif x == '(':
        stuff.append(_parse())
      else:
        stuff.append(x)
    return stuff

  return _parse()[0]

def find(parsed, tag):
  if parsed[0] == tag:
    yield parsed
  for x in parsed[1:]:
    for y in find(x, tag):
      yield y

def unpack(iterable):
   result = []
   for x in iterable:
      if hasattr(x, '__iter__'):
         result.extend(unpack(x))
      else:
         result.append(x)
   return result

        
#p = parse()
#np = find(p, 'PP')
#for x in np:
#  print x