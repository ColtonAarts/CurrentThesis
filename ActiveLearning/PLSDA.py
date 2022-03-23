import spacy
from nltk.corpus import wordnet
from scipy.stats import multinomial
from scipy.stats import bernoulli

nlp = spacy.load('en_core_web_lg')


def format_pos(pos):
    if pos == "a" or pos == "s":
        return "ADJ"
    elif pos == "v":
        return "VERB"
    elif pos == "n":
        return "NOUN"
    elif pos == "r":
        return "ADV"
    else:
        return ''


def format_pos_wordnet(pos):
    if pos == "ADJ":
        return "s"
    elif pos == "VERB":
        return "v"
    elif pos == "NOUN":
        return "n"
    elif pos == "ADV":
        return "r"
    else:
        return ''


def base_PLSDA(sentence_original, target_pos={"ADJ", "ADV", "NOUN"}, probability_ben=0.5, strat="ST", sim_value=0.7, egi=5):
    sentence = nlp(sentence_original)
    final_results = set()
    seen_words = set()
    if strat == "ST":
        lst = list()
        syn_word_dict = dict()
        possible_combinations = 1
        for word in sentence:
            if word.pos_ in target_pos and not word.is_stop and word.text not in seen_words:
                lst.append(word)
                seen_words.add(word.text)
                synsets = wordnet.synsets(word.text)
                possible_synonyms = set()
                for syn in synsets:
                    for l in syn.lemmas():
                        synonym = nlp(l.name())
                        similarity = word.similarity(synonym)
                        if word.pos_ == format_pos(syn.pos()) and similarity > sim_value:
                            possible_synonyms.add(l.name())
                if word.text in possible_synonyms:
                    possible_synonyms.remove(word.text)
                syn_word_dict[word] = possible_synonyms
                possible_combinations *= (len(possible_synonyms) + 1)
        possible_combinations -= 1
        # print(possible_combinations)
        if possible_combinations > egi:
            # print("if")
            seen_before = False
            seen_second = False
            while len(final_results) < egi:
                print("while 1")
                # print(syn_word_dict)
                # print(final_results)
                replacement_syns = dict()
                ben_prob = bernoulli.rvs(probability_ben, size=len(lst))
                while len(set(ben_prob)) < 2 and len(lst) > 1:
                    ben_prob = bernoulli.rvs(probability_ben, size=len(lst))
                if len(lst) == 2 and seen_before and seen_second:
                    ben_prob = [1, 1]
                for index in range(len(ben_prob)):
                    # print(ben_prob)
                    # print("for 1")
                    if len(lst) == 2:
                        if ben_prob[0] == 1:
                            seen_before = True
                        if seen_before and ben_prob[1] == 1:
                            seen_second = True
                    if ben_prob[index] == 1 and len(syn_word_dict[lst[index]]) > 0:
                        word = lst[index]
                        possible_synonyms = syn_word_dict[word]
                        final_replacements = list()
                        possible_synonyms = list(possible_synonyms)
                        if len(possible_synonyms) > 0:
                            array = list()
                            for ele in possible_synonyms:
                                array.append(1 / len(possible_synonyms))
                            cat = multinomial.rvs(1, array)
                            for index in range(len(cat)):
                                if cat[index] == 1:
                                    final_replacements.append(possible_synonyms[index])
                        replacement_syns[word.text] = final_replacements
                changed_sentence = sentence_original
                for ele in replacement_syns.keys():
                    changed_sentence = changed_sentence.replace(ele, replacement_syns[ele][0], 1)
                final_results.add(changed_sentence)
        else:
            replacement_syns = dict()
            # print(lst)
            # print(syn_word_dict)
            for word in lst:
                for ele in syn_word_dict[word]:
                    changed_sentence = sentence_original
                    changed_sentence = changed_sentence.replace(word.text, ele)
                    final_results.add(changed_sentence)
            # print(len(final_results))
            while len(final_results) < possible_combinations:
                print("while")
                print(possible_combinations)
                print(sentence)
                print(lst)
                to_add = list()
                for sentence in final_results:
                    for word in lst:
                        for ele in syn_word_dict[word]:
                            # print("for")
                            changed_sentence = sentence
                            if word.text in changed_sentence.split(" "):
                                changed_sentence = changed_sentence.replace(word.text, ele,1)
                                if changed_sentence != sentence:
                                    to_add.append(changed_sentence)
                for ele in to_add:
                    final_results.add(ele)
                print(len(final_results))

            # for word in lst:

            #     possible_synonyms = syn_word_dict[word]
            #     final_replacements = list()
            #     possible_synonyms = list(possible_synonyms)
            #     if len(possible_synonyms) > 0:
            #         for synanym in possible_synonyms:
            #             final_replacements.append(synanym)
            #     replacement_syns[word.text] = final_replacements
            # changed_sentence = sentence_original
            # for ele in replacement_syns.keys():
            #     changed_sentence = changed_sentence.replace(ele, replacement_syns[ele][0].replace("_", " "))
            # final_results.add(changed_sentence)
    elif strat == "SFS":
        #TODO Fix
        syn_word_dict = dict()
        possible_combinations = 1
        for word in sentence:
            if word.pos_ in target_pos and not word.is_stop:
                synsets = wordnet.synsets(word.text)
                possible_synonyms = set()

                for syn in synsets:
                    for l in syn.lemmas():
                        synonym = nlp(l.name())
                        similarity = word.similarity(synonym)
                        if word.pos_ == format_pos(syn.pos()) and similarity > sim_value and l.name() != word.text:
                            possible_synonyms.add((l.name(), similarity))
                if word.text in possible_synonyms:
                    possible_synonyms.remove(word.text)
                syn_word_dict[word] = possible_synonyms
                possible_combinations *= (len(possible_synonyms) + 1)
            print(syn_word_dict)
            if possible_combinations > egi:
                lst = list()
                # for key in syn_word_dict.keys():
                #     if len(lst) == 0:
                #         for syn in syn_word_dict[key]:
                #             second_list = list()
                #             second_list.append(str(key) + ":" + str(syn[0]))
                #             second_list.append(syn[1])
                #             lst.append(second_list)
                #         lst.append([str(key) + ":" + str(key), 1])
                #     else:
                #         new_list = list()
                #         for syn in syn_word_dict[key]:
                #             for ele in lst:
                #                 new_list.append([ele[0] + "," + str(key) + ":" + str(syn[0]), ele[1] + syn[1]])
                #             new_list.append([lst[-1][0] + "," + str(key) + ":" + str(key), lst[-1][1] + 1])
                #         lst = new_list
                print(lst)
            else:
                print("else")

    return final_results


# doc = "Without Shakespear's eloquent language, the update is dreary and sluggish"
# doc = "a great script brought down by lousy direction"
# doc = "This is a simple sentence to work with"
# nlp_doc = nlp(doc)
#
# for word in nlp_doc:
#     print("(" + word.text + "," + word.pos_ + ")", end=" ")
# print()
# thing = base_PLSDA(doc, strat="ST", egi=80)
# print(thing)
# for ele in sorted(thing):
#     print(ele)

# out_file = open("training_25_PLSDA_7_4.csv", "w+", encoding="utf8")
# test_data = open("training_25.csv", encoding="utf8").read().split("\n")
# for line in test_data:
#     out_file.write(line + "\n")
#     line = line.split(",")
#     print(line[0])
#     extra_sentences = base_PLSDA(line[0], strat="ST")
#     for sentence in extra_sentences:
#         out_file.write(sentence.replace("_", " ") + "," + line[1] + "\n")
