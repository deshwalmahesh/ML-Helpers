import re 
from nltk.stem import SnowballStemmer,WordNetLemmatizer
from nltk.corpus import stopwords 
import sys
import nltk


nltk.download('stopwords')
nltk.download('wordnet')
stemmer = SnowballStemmer('english')
lemmetizer = WordNetLemmatizer()
ver_37 = float(sys.version[:3])>=3.7


stop_words = set(stopwords.words('english')+['diag','diagram','ill','illustration','fig','figure',
'iit','cbse','jee','uptu','array','i','ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii']) 


special_char_mapping = {
  "†": "+","∞": "infinity","×": "*","°": "degree","–": "-","~": "negation","・": "*","—": "-",
  "√": "square root","•": "*","」": "floor",
  }

# yellow
mis_spelled = {'tany': 'tan y', 'thea': 'theta', 'bycm': 'by centimeter', 'wihcha': 'which a', 
'trainglea': 'trianlge a', 'liner': 'linear', 'whena': 'when a', 'thenn': 'then n', 
'inx': 'in x', 'acos': 'a cos', 'tano': 'tan o', 'aarray': 'an array', 'andv': 'and v', 'ande': 'and e', 
'givena': 'given a', 'thenc': 'then c', 'fory': 'for y', 'thatc': 'that c', 'withp': 'with p', 'ofi': 'of i', 
'letr': 'let r', 'thecost': 'the cost', 'findk': 'find k', 'andh': 'and h', 'andz': 'and z', 'findb': 'find b', 
'asa': 'as a', 'letp': 'let p', 'ifc': 'if c', 'wherei': 'where i', 'areacm': 'area centimeter', 'findm': 'find m', 
'ife': 'if e', 'ifz': 'if z', 'onr': 'on r', 'asf': 'as f', 'isy': 'is y', 'isp': 'is p', 'bea': 'be a', 
'equationa': 'equation a', 'ands': 'and s', 'byy': 'by y', 'liney': 'line y', 'ofc': 'of c', 'atp': 'at p', 
'ofn': 'of n', 'thatcm': 'that centimeter', 'tocm': 'to centimeter', 'theny': 'then y', 'ifr': 'if r', 'parabolay': 'parabola y',
'ifb': 'if b', 'sas': 's as', 'ofcosec': 'of cosec', 'equationy': 'equation y', 'thenk': 'then k', 
'sidescm': 'sides centimeter', 'sol': 'solution', 'tob': 'to b', 'cotx': 'cot x', 'ina': 'in a', 'ing': 'in g', 
'froma': 'from a', 'fora': 'for a', 'byr': 'by r', 'ofk': 'of k', 'sina': 'sin a', 'ando': 'and o', 
'thatb': 'that b', 'thatx': 'that x', 'arrayarray': 'array', 'arraya': 'array a', 'ifd': 'if d', 
'pointsp': 'points p', 'planex': 'plane x', 'isx': 'is x', 'vectori': 'vector i', 'thenb': 'then b', 
'incm': 'in centimeter', 'ifi': 'if i', 'cosa': 'cos a', 'cose': 'cos e', 'cosx': 'cos x', 'sino': 'sin o', 
'whenx': 'when x', 'ifm': 'if m', 'relationr': 'relation r', 'vectorsi': 'vectors i', 'linesx': 'lines x', 
'thatarray': 'that array', 'tox': 'to x', 'findp': 'find p', 'ofb': 'of b', 'andarray': 'and array', 
'sidecm': 'side centimeter', 'betweena': 'between a', 'heightcm': 'height centimeter', 'lengthcm': 'length centimeter', 
'ofy': 'of y', 'bya': 'by a', 'ofarray': 'of array', 'findf': 'find f', 'andq': 'and q', 'ifcm': 'if centimeter', 
'findx': 'find x', 'matrixa': 'matrix a', 'pointp': 'point p', 'monthlyarray': 'monthly array', 
'carray': 'c array', 'plancml': 'plan centimeter l', 'adjecnt': 'adjacent', 'floorm': 'floor m', 
'rectanglein': 'rectangle in', 'whichn': 'which n', 'thann': 'than n', 'simpify': 'simplify', 
'productba': 'product ba', 'rotat': 'rotate', 'whenu': 'when', 'thatqy': 'that qy', 'ifmm': 'if millimeter', 
'papercm': 'paper centimeter', 'neeed': 'need', 'emaining': 'remaining', 'ammmm': 'am', 
'ploynomars': 'polynomers', 'ifxx': 'if xx', 'forcef': 'force f', 'cm': 'centimetre', 'isa': 'is a', 
'ofcm': 'of centimeter', 'mm': 'millimeter', 'iscm': 'is centimeter', 'cannot': 'can not', 'massm': 'mass m', 
'ofa': 'of a', 'toc': 'to c', 'radiuscm': 'radius centimeter', 'metre': 'meter', 'andb': 'and b', 
'andy': 'and y', 'andcm': 'and centimeter', 'atcm': 'at centimeter', 'ofkm': 'of kilometer', 
'andc': 'and c', 'andd': 'and d', 'fromc': 'from c', 'lengthm': 'length m', 'lengthl': 'length l',
'functionf': 'function f', 'hr': 'hour', 'velocityv': 'velocity v', 'speeds': 'speed s', 
'distanced': 'distance d', 'vectorv': 'vector v', 'gm': 'gram', 'kg': 'kilogram', 'cmcm': 'centimeter', 
'toa': 'to a', 'ofkg': 'of kilogram', 'resistancer': 'resistance r', 'voltagev': 'voltage v', 
'andr': 'and r', 'fielde': 'field e',"cm":"centimeter", 'cms': 'centimeter', 'distancecm': 'distance centimeter',
'ismm': 'is millimeter', 'pm': 'picometer', 'nm': 'nanometer', 'fieldb': 'field', 'akg': 'kilogram', 
'isn': 'is n', 'njunction': 'n junction', 'anticlockwise': 'anti clockwise', 'ifa': 'if a',
'ifan': 'if an', 'area cm': 'area centimeter',"ns": "nano second", "placedcm": "placed centimeter", 
"isg": "is g", "thena": "then a", "andi": "and i", "mg": "milligram", "takeg": "take g", 
"currenti": "current i", "byx": "by x", "velocitym": "velocity m", "inm": "in meter", 
"inml": "in milliliter", "iskg": "is kilogram", "anda": "and a", "hz": "hertz", 
"frequencyhz": "frequency hertz", "optionsa": "option", "coeficient": "coefficient", "andn": "and n",
"mol": "mole", "andkg": "and kilogram", "cant": "can not", "becm": "be centimeter", "pointa": "point a",
"andf": "and f", "speedv": "speed v", "chargesq": "charge q", "vectorsa": "vector", "ofx": "of x", 
"iskm": "is kilometer", "curvaturecm": "curvature centimeter", "vectora": "vector a", "doesn": "does", 
"timet": "time t", "atkm": "at kilometer", "isms": "is millisecond", "heighth": "height h", 
"equationx": "equation x", "andt": "and t", "pointcm": "point centimeter", "ishz": "is hertz", 
"khz": "kilohertz", "andp": "and p", "amplitudecm": "amplitude centimeter", "bymm": "by millimeter", 
"pointb": "point b", "wherex": "where x", "radiusmm": "radius millimeter", "alongx": "along x", 
"biogas": "bio gas"}

math_latex = {'perp': 'perpendicular', 'circ': 'circle from sets', 'tan': 'tan', 'sin': 'sine', 
'tilde': 'tilde', 'congruent': 'congruent', 'series': 'math series', 'exp': 'exponential', 
'widehat': 'vector', 'emptyset': 'empty set', 'conic': 'conic section', 'prime': 'differentiation prime', 
'nsubseteq': 'set not equal to', 'acos': 'cos', 'cosce': 'cosecant', 'sets': 'sets', 'cdot': 'dot product', 
'rangle': 'angled bracket', 'beta': 'beta', 'cosines': 'cosines', 'imath': 'dotless i', 'sqrt': 'square root',
'psi': 'greek psi', 'circle': 'circle', 'epsilon': 'epsilon', 'triangle': 'triangle', 
'arccos': 'inverse cosine', 'diameter': 'diameter', 'disc': 'disc', 'subseteq': 'subset equals to', 
'otimes': 'cross product', 'square': 'square', 'rational': 'rational', 'imaginary': 'imaginary number', 
'constant': 'constant', 'not': 'negation not', 'varepsilon': 'varepsilon', 'alpha': 'alpha', 
'nor': 'logical nor', 'supset': 'super set', 'mu': 'mu', 'angles': 'angle', 'arcsin': 'inverse sine', 
'gamma': 'gamma', 'zeta': 'zeta', 'bigcup': 'set union', 'circles': 'circle', 'theta': 'theta', 
'infty': 'infinity', 'overline': 'bar', 'odot': 'inner dot product', 'omega': 'omega', 
'longleftrightarrow': 'if and only if', 'upsilon': 'upsilon', 'lambda': 'lambda', 'dots': 'series dots', 
'pi': 'pi', 'neq': 'not equal to', 'hline': 'horizontal line', 'cos': 'cos', 'underline': 'underline', 
'chi': 'chi', 'iota': 'iota', 'sec': 'secant', 'vdots': 'vertical dots', 'arctan': 'inverse tangent', 
'bar': 'bar', 'cdots': 'centered dot', 'leqslant': 'less than or equal to', 'frac': 'fraction', 
'bigcap': 'set intersection', 'set': 'set', 'array': 'matrix', 'vec': 'vector symbol', 
'overbrace': 'over brace', 'parallel': 'parallel', 'sinh': 'hyperbolic sine', 'mathbb': 'blackboard bold',
'tanh': 'hyperbolic tangent', 'cot': 'cotangent', 'differentiate': 'differentiate', 'cosec': 'cosecant', 
'matrices': 'matrix', 'transpose': 'transpose', 'notin': 'set not in', 'jmath': 'dotless j ', 
'propto': 'propto', 'cong': 'congruent', 'rfloor': 'right floor', 'log': 'logarithm', 
'ldots': 'lower dots', 'sim': 'asymptotic equality', 'geq': 'greater than or equal to', 'delta': 'delta',
'cosine': 'cosine', 'div': 'division', 'vartheta': 'theta', 'sqsubseteq': 'subset equal to', 
'varphi': 'phi', 'nabla': 'nabla', 'oplus': 'exclusive or sets', 'arccot': 'inverse cotangent', 
'root': 'root of', 'leq': 'less than or equal to', 'matrix': 'matrix', 'leftrightarrow': 'if and only if',
'simeq': 'approximately equal', 'prod': 'pi product', 'geqslant': 'greater than or equal to', 
'nsupseteq': 'superset not equal to', 'nu': 'nu', 'phi': 'phi', 'lfloor': 'left floor', 'cup': 'set union',
'sigma': 'sigma', 'det': 'determinant', 'tau': 'tau', 'cap': 'set intersection', 'rho': 'rho'}


keep_spec_char = '+-=<>%*/^'
discard = ['_','i','ii','iii','iv','v','vi','vii','viii','ix','x','xi','xii']

def is_ascii(s):
    return all(ord(c) < 128 for c in s)
    

def filter_correct(sentence,use_lemmetization=False,use_stemming=False,
remove_single_words=False,remove_numbers=False,change_nums_to_strs=False,remove_stopwords=False,remove_special_char=False):
  '''
  Correct the spellings and change special characters names
  args:
    change_nums_to_strs: Change 2, 45, 3434 all to "number"
    remove_single_words: remove x, y, z, i, etc
  '''
  splitted = sentence.split(' ')
  dummy_list = []
  for token in splitted:
    token = token.strip()
    if token.isalpha() and (token.isascii() if ver_37 else is_ascii(token) ) and token not in discard: # if token is made up of only alphabets and within ASCII range
      if token in mis_spelled: # correct the word. There will be 1 or more resulting tokens so correct each one
        new_tokens = mis_spelled[token]

        for token in new_tokens.split(' '):
          result = preprocess(token,use_lemmetization=use_lemmetization,use_stemming=use_stemming,
          remove_single_words=remove_single_words,remove_numbers=remove_numbers,
          change_nums_to_strs=change_nums_to_strs,remove_stopwords=remove_stopwords,
          remove_special_char=remove_special_char)

          dummy_list.append(result)

      else: # if it is not in mispelled
        if remove_single_words and len(token)==1: #remove x,y,i,v etc
          continue

        if remove_stopwords and token in stop_words: # if token is a stopword
          continue

        if use_lemmetization: # if lemmetize only 
          token = lemmetizer.lemmatize(token)

        if use_stemming: # stemming only
          token = stemmer.stem(token)
          
        dummy_list.append(token)


    elif token.isdigit(): # if token is digit
      if remove_numbers:
        continue
      if change_nums_to_strs: # change number to "number"
        dummy_list.append('number')
      else:
        dummy_list.append(token)
      
    else: # if it is special char
      if remove_special_char:
        continue
      else:
        if token in keep_spec_char: # if it is + - etc
          dummy_list.append(token)
        elif token in special_char_mapping: # if it is infinity, degree etc
          dummy_list.append(special_char_mapping[token])

  return ' '.join(dummy_list)


def remove_single_word_num(sent):
  '''
  Remove numbers and words of single length such as "x + 23 y - abc" will become "+ - abc"
  '''
  dummy_list = []
  
  for token in sent.split():
    if (not token.isdigit()) and ((token.isalpha() and len(token)>1) or (not token.isalnum())):
      dummy_list.append(token)   
  
  return ' '.join(dummy_list)


def insert_spaces(sentence):
  '''
  Add a space around special characters, number and digits. So "2x+y -1/3x" becomes: "2 x + y - 1 / 3 x"
  '''
  dummy_list = []
  splitted_sent = list(sentence)
    
  for i in range(len(splitted_sent)-1):
    dummy_list.append(splitted_sent[i])
    
    if splitted_sent[i].isalpha(): # if it is an alphabet
      if splitted_sent[i+1].isdigit() or (not splitted_sent[i+1].isalnum()):
        dummy_list.append(' ')
    
    elif splitted_sent[i].isdigit(): # if it is a number
      if splitted_sent[i+1].isalpha() or (not splitted_sent[i+1].isalnum()):
        dummy_list.append(' ')
        
    elif (not splitted_sent[i].isalnum()) and (splitted_sent[i] not in [' ','\\']): # if it is a special char but not ' ' already
      if splitted_sent[i+1].isalnum():
        dummy_list.append(' ')
        
  dummy_list.append(splitted_sent[-1])
  
  return ''.join(dummy_list)


UPDATE_CODE = True
def clean_latex():
    t = []
    if isinstance(sentence,str):
        sentence = insert_spaces(sentence)
        for token in sentence.split('\\'):
            for tok in token.split(' '):
                token = tok.strip().lower()
                if (token.isascii() and token.isalpha()):
                    t.append(token) 
    return ' '.join(t)



def preprocess(sentence,use_lemmetization=False,use_stemming=False,remove_single_words=False,
remove_numbers=False,change_nums_to_strs=False,remove_stopwords=False,remove_special_char=False):
  
  assert not(remove_numbers and change_nums_to_strs), "Use either one of Remove Number or Change num to string"
  
  # convert the characters into lower case
  a = sentence.lower()

  # remomve newline character
  a = re.sub("\\n", " ", a)

  # remove the pattern [ whatever here ]. Use { } or  ( ) in place of [ ] in regex
  a = re.sub(r"\[(.*?)\]",' ',a)

  # remove abbrevationns like I.I.T. , JEE, J.e.e. etc 
  a = re.sub(r"(?<!\S)(?:(?:[A-Za-z]\.){3,}|[A-Z]{3,})(?!\S)",' ',a)

  # remove Questions beginners Q5. 5. question 5. 
  a = re.sub(r"^[\w]+(\s|\.)(\s|\d+(\.*(\d+|\s)))\s*", " ", a)

  # remove MathPix markdown starting from \( and ending at \) while preserving data inside \text { preserve this }
  a = re.sub(r'\s*\\+\((.*?)\\+\)', lambda x: " ".join(re.findall(r'\\text\s*{([^{}]*)}', x.group(1))), repr(a))

  # remove options from questions i.e character bounded by () given there is no spacing inside ()
  a = re.sub(r"\s*\([^)\s]*\)\s*", " ", a)

  # remove data inside {} -> at max 2 characters {q.}, {5.}
  a = re.sub(r"{.{0,2}}", " ", a)

  # Insert spaces among spec chars, digits and nums
  a = insert_spaces(a)

  # remove whatever comes after \\ double or single slashes except space 
  a = re.sub(r"(\\[^ ]+)",' ',a)


  # remove strings which are not ASCII, correct spellings, map special characters
  a = filter_correct(a,use_lemmetization=use_lemmetization,
  use_stemming=use_stemming,remove_single_words=remove_single_words,remove_numbers=remove_numbers,
  change_nums_to_strs=change_nums_to_strs,remove_stopwords=remove_stopwords,remove_special_char=remove_special_char)


  # remomve newline character inserted by raw  string in the removal of Mathpix
  a = re.sub("\n", " ", a)

  # remove any repeating special character (more than one times) except space.  So it'll remove ._  ++ etc +-= except spaces
  a = re.sub(r"([^a-zA-Z0-9 ]{2,})",' ',a)

  # remove repeated space if there is any
  a = re.sub(r"\s+", " ", a)
  
  return a
