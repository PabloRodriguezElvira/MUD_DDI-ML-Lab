#! /usr/bin/python3

import sys
from os import listdir

from xml.dom.minidom import parse

from deptree import *
#import patterns


# Clue verbs associated with each DDI type (lemma form, lowercase)
CLUE_VERBS = {
    "effect":    {"increase", "decrease", "reduce", "enhance", "potentiate",
                  "block", "cause", "produce", "affect", "alter", "raise",
                  "lower", "elevate", "augment", "attenuate", "impair",
                  "improve", "worsen", "intensify", "diminish"},
    "mechanism": {"inhibit", "induce", "metabolize", "bind", "compete",
                  "displace", "absorb", "distribute", "eliminate", "excrete",
                  "oxidize", "conjugate", "glucuronidate", "hydrolyze",
                  "acetylate", "interact"},
    "advise":    {"avoid", "recommend", "monitor", "administer", "consider",
                  "warn", "caution", "contraindicate", "adjust", "discontinue",
                  "replace", "substitute", "use", "combine"},
    "int":       {"interact", "co-administer", "co-prescribe"},
}

# flat map: lemma -> category
_VERB_TO_CAT = {v: cat for cat, verbs in CLUE_VERBS.items() for v in verbs}


def _add_clue_verb_feats(feats, lemma, position):
    """Emit clue-verb features if lemma is a known clue verb."""
    cat = _VERB_TO_CAT.get(lemma)
    if cat is not None:
        feats.add(f"clue_verb_{position}={lemma}")   # verb identity + position
        feats.add(f"clue_cat_{position}={cat}")       # category + position
        feats.add(f"has_clue_verb_{position}=True")   # binary presence


## -------------------
## -- Convert a pair of drugs and their context in a feature vector

def extract_features(tree, entities, e1, e2) :
   feats = set()

   # get head token for each gold entity
   tkE1 = tree.get_fragment_head(entities[e1]['start'],entities[e1]['end'])
   tkE2 = tree.get_fragment_head(entities[e2]['start'],entities[e2]['end'])

   if tkE1 is not None and tkE2 is not None:

      # entity types
      feats.add("e1type=" + entities[e1].get('type',''))
      feats.add("e2type=" + entities[e2].get('type',''))
      feats.add("pairtype=" + entities[e1].get('type','') + "_" + entities[e2].get('type',''))

      # tokens BEFORE e1: word, lemma, lemma+PoS, verbs
      for tk in range(1, tkE1):
         lemma = tree.get_lemma(tk).lower()
         word  = tree.get_word(tk)
         tag   = tree.get_tag(tk)
         feats.add("lb1=" + lemma)
         feats.add("wb1=" + word)
         feats.add("lpb1=" + lemma + "_" + tag)
         if tag.startswith('V'):
            feats.add("verb_before=" + lemma)
            _add_clue_verb_feats(feats, lemma, "before")

      # tokens IN BETWEEN e1 and e2: word, lemma, lemma+PoS, verbs
      eib = False
      for tk in range(tkE1+1, tkE2):
         lemma = tree.get_lemma(tk).lower()
         word  = tree.get_word(tk)
         tag   = tree.get_tag(tk)
         feats.add("lib=" + lemma)
         feats.add("wib=" + word)
         feats.add("lpib=" + lemma + "_" + tag)
         if tag.startswith('V'):
            feats.add("verb_between=" + lemma)
            _add_clue_verb_feats(feats, lemma, "between")
         if tree.is_entity(tk, entities):
            eib = True

      feats.add("eib=" + str(eib))

      # tokens AFTER e2: word, lemma, lemma+PoS, verbs
      for tk in range(tkE2+1, tree.get_n_nodes()):
         lemma = tree.get_lemma(tk).lower()
         word  = tree.get_word(tk)
         tag   = tree.get_tag(tk)
         feats.add("la2=" + lemma)
         feats.add("wa2=" + word)
         feats.add("lpa2=" + lemma + "_" + tag)
         if tag.startswith('V'):
            feats.add("verb_after=" + lemma)
            _add_clue_verb_feats(feats, lemma, "after")

      # path features
      lcs = tree.get_LCS(tkE1, tkE2)

      feats.add("LCSpos="  + tree.get_tag(lcs))
      feats.add("LCSlema=" + tree.get_lemma(lcs).lower())

      path1_nodes = tree.get_up_path(tkE1, lcs)
      path2_nodes = tree.get_down_path(lcs, tkE2)

      if path1_nodes is not None and path2_nodes is not None:
         # paths encoded with lemma+relation
         path1_str = "<".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path1_nodes])
         path2_str = ">".join([tree.get_lemma(x)+"_"+tree.get_rel(x) for x in path2_nodes])
         lcs_str   = tree.get_lemma(lcs)+"_"+tree.get_rel(lcs)
         feats.add("path1=" + path1_str)
         feats.add("path2=" + path2_str)
         feats.add("path="  + path1_str + "<" + lcs_str + ">" + path2_str)

         # paths encoded with PoS+relation (more generalizable)
         path1_pos = "<".join([tree.get_tag(x)+"_"+tree.get_rel(x) for x in path1_nodes])
         path2_pos = ">".join([tree.get_tag(x)+"_"+tree.get_rel(x) for x in path2_nodes])
         lcs_pos   = tree.get_tag(lcs)+"_"+tree.get_rel(lcs)
         feats.add("path1pos=" + path1_pos)
         feats.add("path2pos=" + path2_pos)
         feats.add("pathpos="  + path1_pos + "<" + lcs_pos + ">" + path2_pos)

   return feats


## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  extract_features targetdir
## --
## -- Extracts feature vectors for DD interaction pairs from all XML files in target-dir
## --

# directory with files to process
datadir = sys.argv[1]

# process each file in directory
for f in listdir(datadir) :

    # parse XML file, obtaining a DOM tree
    tree = parse(datadir+"/"+f)

    # process each sentence in the file
    sentences = tree.getElementsByTagName("sentence")
    for s in sentences :
        sid = s.attributes["id"].value   # get sentence id
        stext = s.attributes["text"].value   # get sentence text
        # load sentence entities
        entities = {}
        ents = s.getElementsByTagName("entity")
        for e in ents :
           id = e.attributes["id"].value
           offs = e.attributes["charOffset"].value.split("-")
           etype = e.attributes["type"].value if e.hasAttribute("type") else ""
           entities[id] = {'start': int(offs[0]), 'end': int(offs[-1]), 'type': etype}

        # there are no entity pairs, skip sentence
        if len(entities) <= 1 : continue

        # analyze sentence
        analysis = deptree(stext)

        # for each pair in the sentence, decide whether it is DDI and its type
        pairs = s.getElementsByTagName("pair")
        for p in pairs:
            # ground truth
            ddi = p.attributes["ddi"].value
            if (ddi=="true") : dditype = p.attributes["type"].value
            else : dditype = "null"
            # target entities
            id_e1 = p.attributes["e1"].value
            id_e2 = p.attributes["e2"].value
            # feature extraction

            feats = extract_features(analysis,entities,id_e1,id_e2) 
            # resulting vector
            if len(feats) != 0:
              print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t")

