import random
import re
import math
from collections import namedtuple
import dynet as dy
from datetime import datetime
import pickle
import logging.config
import numpy as np

logging.config.fileConfig('resources/logging.ini')
logger = logging.getLogger(__file__)


class TrMorphTagger(object):

    # HYPER-PARAMETERS
    CHAR_EMBEDDING_SIZE = 64
    ENC_STATE_SIZE = 512
    DEC_STATE_SIZE = 256

    # STATIC VARIABLES
    SENTENCE_BEGIN_TAG = "<s>"
    SENTENCE_END_TAG = "</s>"

    # COMPILED REGEX PATTERNS
    analysis_regex = re.compile(r"^([^\\+]*)\+(.+)$", re.UNICODE)
    tag_seperator_regex = re.compile(r"[\\+\\^]", re.UNICODE)
    split_root_tags_regex = re.compile(r"^([^\\+]+)\+(.+)$", re.IGNORECASE)

    # WORD STRUCT
    WordStruct = namedtuple("WordStruct", ["surface_word", "roots", "tags"])

    @classmethod
    def _create_vocab(cls, sentences):
        logger.info("Building Vocabulary...")
        char2id = dict()
        char2id["<"] = len(char2id)
        char2id["/"] = len(char2id)
        char2id[">"] = len(char2id)
        char2id[TrMorphTagger.SENTENCE_BEGIN_TAG] = len(char2id)
        char2id[TrMorphTagger.SENTENCE_END_TAG] = len(char2id)
        for sentence in sentences:
            for word in sentence:
                for ch in word.surface_word:
                    if ch not in char2id:
                        char2id[ch] = len(char2id)
                for root in word.roots:
                    for ch in root:
                        if ch not in char2id:
                            char2id[ch] = len(char2id)
                for tags in word.tags:
                    for tag in tags:
                        if tag not in char2id:
                            char2id[tag] = len(char2id)
        id2char = {v: k for k, v in char2id.items()}
        logger.info("Done.")
        return char2id, id2char

    @classmethod
    def _encode(cls, tokens, vocab):
        return [vocab[token] for token in tokens if token in vocab]

    @classmethod
    def _embed(cls, token, char_embedding_table):
        return [char_embedding_table[ch] for ch in token]

    def __init__(self, train_from_scratch=True,
                 train_data_path="data/data.train.txt",
                 dev_data_path="data/data.dev.txt", test_data_paths=["data/data.test.txt"],
                 model_file_name=None,
                 case_sensitive=False):

        self.case_sensitive = case_sensitive

        if train_from_scratch:
            assert train_data_path
            assert len(test_data_paths) > 0
            self.train = self.load_data(train_data_path)
            if dev_data_path:
                self.dev = self.load_data(dev_data_path)
            else:
                self.dev = None
            self.test_paths = test_data_paths
            self.tests = []
            for test_path in self.test_paths:
                self.tests.append(self.load_data(test_path))

            self.char2id, self.id2char = self._create_vocab(self.train)

            if not self.dev:
                train_size = int(math.floor(0.99 * len(self.train)))
                self.dev = self.train[train_size:]
                self.train = self.train[:train_size]

            self.model = dy.Model()
            self.trainer = dy.SimpleSGDTrainer(self.model, learning_rate=1.6)
            self.CHARS_LOOKUP = self.model.add_lookup_parameters((len(self.char2id),
                                                                  TrMorphTagger.CHAR_EMBEDDING_SIZE))

            self.individual_word_rnn = dy.LSTMBuilder(1,
                                                      TrMorphTagger.CHAR_EMBEDDING_SIZE,
                                                      TrMorphTagger.ENC_STATE_SIZE,
                                                      self.model)
            self.individual_word_rnn.set_dropout(0.3)

            self.tag_rnn = dy.LSTMBuilder(1,
                                          TrMorphTagger.CHAR_EMBEDDING_SIZE,
                                          TrMorphTagger.ENC_STATE_SIZE,
                                          self.model)
            self.tag_rnn.set_dropout(0.3)

            self.surface_rnn = dy.LSTMBuilder(1,
                                              TrMorphTagger.CHAR_EMBEDDING_SIZE,
                                              TrMorphTagger.ENC_STATE_SIZE,
                                              self.model)
            self.surface_rnn.set_dropout(0.3)

            self.fwd_context_rnn = dy.LSTMBuilder(1,
                                                  TrMorphTagger.ENC_STATE_SIZE,
                                                  TrMorphTagger.ENC_STATE_SIZE,
                                                  self.model)
            self.fwd_context_rnn.set_dropout(0.3)

            self.bwd_context_rnn = dy.LSTMBuilder(1,
                                                  TrMorphTagger.ENC_STATE_SIZE,
                                                  TrMorphTagger.ENC_STATE_SIZE,
                                                  self.model)
            self.bwd_context_rnn.set_dropout(0.3)

            self.DEC_RNN = dy.LSTMBuilder(2,
                                          TrMorphTagger.CHAR_EMBEDDING_SIZE,
                                          TrMorphTagger.ENC_STATE_SIZE,
                                          self.model)
            self.DEC_RNN.set_dropout(0.3)

            # context fully connected layer
            self.context_w = self.model.add_parameters((TrMorphTagger.ENC_STATE_SIZE, TrMorphTagger.ENC_STATE_SIZE * 2))
            self.context_b = self.model.add_parameters(TrMorphTagger.ENC_STATE_SIZE)

            # project the rnn output to a vector of VOCAB_SIZE length
            self.output_w = self.model.add_parameters((len(self.char2id), TrMorphTagger.ENC_STATE_SIZE))
            self.output_b = self.model.add_parameters((len(self.char2id)))

            self.train_model(model_name=model_file_name)
        else:
            logger.info("Loading Pre-Trained Model")
            assert model_file_name
            self.load_model(model_file_name)

    def _get_tags_from_analysis(self, analysis):
        if analysis.startswith("+"):
            return self.tag_seperator_regex.split(analysis[2:])
        else:
            return self.tag_seperator_regex.split(self.analysis_regex.sub(r"\2", analysis))

    def _get_root_from_analysis(self, analysis):
        if analysis.startswith("+"):
            return "+"
        else:
            return self.analysis_regex.sub(r"\1", analysis)

    def _get_pos_from_analysis(self, analysis):
        tags = self._get_tagsstr_from_analysis(analysis)
        if "^" in tags:
            tags = tags[tags.rfind("^") + 4:]
        return tags.split("+")[0]

    def _get_tagsstr_from_analysis(self, analysis):
        if analysis.startswith("+"):
            return analysis[2:]
        else:
            return self.split_root_tags_regex.sub(r"\2", analysis)

    def load_data(self, file_path, max_sentence=10):
        logger.info("Loading data from {}".format(file_path))
        sentence = []
        sentences = []
        with open(file_path, 'r', encoding="UTF-8") as f:
            for line in f:
                trimmed_line = line.strip(" \r\n\t")
                if trimmed_line.startswith("<S>") or trimmed_line.startswith("<s>"):
                    sentence = []
                elif trimmed_line.startswith("</S>") or trimmed_line.startswith("</s>"):
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        if len(sentences) > max_sentence:
                            return sentences
                elif len(trimmed_line) == 0 or "<DOC>" in trimmed_line or trimmed_line.startswith(
                        "</DOC>") or trimmed_line.startswith(
                        "<TITLE>") or trimmed_line.startswith("</TITLE>"):
                    pass
                else:
                    parses = re.split(r"[\t ]", trimmed_line)
                    surface = parses[0]
                    analyzes = parses[1:]
                    roots = [self._get_root_from_analysis(analysis) for analysis in analyzes]
                    tags = [self._get_tags_from_analysis(analysis) for analysis in analyzes]
                    if not self.case_sensitive:
                        surface = TrMorphTagger.lower(surface)
                        roots = [TrMorphTagger.lower(root) for root in roots]
                    current_word = self.WordStruct(surface, roots, tags)
                    sentence.append(current_word)
        logger.info("Done.")
        return sentences

    @staticmethod
    def _run_rnn(init_state, input_vecs):
        s = init_state

        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors

    def _get_word_representations(self, words):
        word_representations = []
        individual_word_rnn_init = self.individual_word_rnn.initial_state()
        for index, word in enumerate(words):
            encoded_surface_word = self._encode(word, self.char2id)
            if len(encoded_surface_word) <= 0:
                continue
            individual_word_embeddings = self._embed(encoded_surface_word, self.CHARS_LOOKUP)
            individual_word_rep = self._run_rnn(individual_word_rnn_init, individual_word_embeddings)
            word_representations.append(individual_word_rep[-1])
        return word_representations

    def _get_context_representations(self, words):
        w = dy.parameter(self.context_w)
        b = dy.parameter(self.context_b)

        # RENEW COMPUTATION GRAPH
        surface_rnn_init = self.surface_rnn.initial_state()

        fwd_context_rnn_init = self.fwd_context_rnn.initial_state()
        bwd_context_rnn_init = self.bwd_context_rnn.initial_state()

        # SURFACE WORD REPRESENTATION
        surface_words_reps = []
        for index, word in enumerate(words):
            encoded_surface_word = self._encode(word, self.char2id)
            if len(encoded_surface_word) <= 0:
                continue
            surface_word_char_embeddings = self._embed(encoded_surface_word, self.CHARS_LOOKUP)
            surface_word_rep = self._run_rnn(surface_rnn_init, surface_word_char_embeddings)
            surface_words_reps.append(surface_word_rep[-1])

        # CONTEXT REPRESENTATIONS
        fwd_context = self._run_rnn(fwd_context_rnn_init, surface_words_reps)
        bwd_context = self._run_rnn(bwd_context_rnn_init, reversed(surface_words_reps))[::1]

        # concatanate the foreward and backword outputs
        context_representations = []
        for i in range(len(fwd_context)):
            if i == 0:
                context_representations.append(dy.concatenate([bwd_context[0],
                                                              dy.zeroes(TrMorphTagger.ENC_STATE_SIZE)]))
            elif i + 1 == len(fwd_context):
                context_representations.append(dy.concatenate([fwd_context[-1],
                                                              dy.zeroes(TrMorphTagger.ENC_STATE_SIZE)]))
            else:
                context_representations.append(dy.concatenate([fwd_context[i-1],
                                                              bwd_context[i+1]]))
        # context_representations = [dy.concatenate([fwd_out, bwd_out])
        #                            for fwd_out, bwd_out in zip(fwd_context, bwd_context)]
        context_representations = [dy.rectify(w * context_representation + b)
                                   for context_representation in context_representations]

        return context_representations

    @staticmethod
    def lower(text):
        text = text.replace("Ü", "ü")
        text = text.replace("Ğ", "ğ")
        text = text.replace("İ", "i")
        text = text.replace("Ş", "ş")
        text = text.replace("Ç", "ç")
        text = text.replace("Ö", "ö")
        return text.lower()

    def _get_probs(self, rnn_output):
        output_w = dy.parameter(self.output_w)
        output_b = dy.parameter(self.output_b)

        probs = dy.softmax(output_w * rnn_output + output_b)
        return probs

    def get_loss(self, sentence):
        dy.renew_cg()
        surface_words = [word.surface_word for word in sentence]
        if not self.case_sensitive:
            surface_words = [TrMorphTagger.lower(word) for word in surface_words]

        context_representations = self._get_context_representations(surface_words)
        word_representations = self._get_word_representations(surface_words)

        root_char_sequences = [word.roots[0] for word in sentence]
        if not self.case_sensitive:
            root_char_sequences = [TrMorphTagger.lower(root) for root in root_char_sequences]

        tag_sequences = [word.tags[0] for word in sentence]

        gold_sequences = [[TrMorphTagger.SENTENCE_BEGIN_TAG] + list(root) + tags + [TrMorphTagger.SENTENCE_END_TAG]
                          for root, tags in zip(root_char_sequences, tag_sequences)]

        probs = []
        losses = []
        tag_rnn_state = self.tag_rnn.initial_state()
        for gold_sequence, context_representation, word_representation \
                in zip(gold_sequences, context_representations, word_representations):

            if tag_rnn_state.output():
                output_encoder_output = word_representation + tag_rnn_state.output()
            else:
                output_encoder_output = word_representation

            rnn_state = self.DEC_RNN.initial_state().set_s([dy.tanh(context_representation),
                                                            dy.tanh(context_representation),
                                                            dy.tanh(output_encoder_output),
                                                            dy.tanh(output_encoder_output)])
            for gold_i in gold_sequence:
                rnn_state = rnn_state.add_input(self.CHARS_LOOKUP[self.char2id[gold_i]])
                p = self._get_probs(rnn_state.output())
                probs.append(p)

            for gold_i in gold_sequence:
                tag_embedding = self.CHARS_LOOKUP[self.char2id[gold_i]]
                tag_rnn_state.add_input(tag_embedding)

            losses += [-dy.log(dy.pick(p, self.char2id[output_char])) for p, output_char in
                       zip(probs, gold_sequence)]
        return dy.esum(losses)

    def generate(self, sentence, max_tag_count=50):
        surface_words = [word.surface_word for word in sentence]
        if not self.case_sensitive:
            surface_words = [TrMorphTagger.lower(word) for word in surface_words]

        context_representations = self._get_context_representations(surface_words)
        word_representations = self._get_word_representations(surface_words)

        predicted_sequences = []
        tag_rnn_state = self.tag_rnn.initial_state()
        for context_representation, word_representation in zip(context_representations, word_representations):
            if tag_rnn_state.output():
                output_encoder_output = word_representation + tag_rnn_state.output()
            else:
                output_encoder_output = word_representation
            predicted_sequence = []
            rnn_state = self.DEC_RNN.initial_state().set_s([context_representation, dy.tanh(context_representation),
                                                            output_encoder_output, dy.tanh(output_encoder_output)])
            predicted_char = TrMorphTagger.SENTENCE_BEGIN_TAG
            counter = 0
            while True:
                counter += 1
                rnn_state = rnn_state.add_input(self.CHARS_LOOKUP[self.char2id[predicted_char]])
                probs = self._get_probs(rnn_state.output())
                predicted_char = self.id2char[probs.npvalue().argmax()]
                if predicted_char == TrMorphTagger.SENTENCE_END_TAG or counter >= max_tag_count:
                    break
                else:
                    predicted_sequence.append(predicted_char)
            for seq_i in predicted_sequence:
                tag_embedding = self.CHARS_LOOKUP[self.char2id[seq_i]]
                tag_rnn_state.add_input(tag_embedding)
            predicted_sequences.append(predicted_sequence)
        return TrMorphTagger._convert_to_morph_analyzer_form(predicted_sequences)

    @staticmethod
    def _convert_to_morph_analyzer_form(sequence_arr):
        res = []
        for sequence in sequence_arr:
            word_result = []
            for x in sequence:
                if x == TrMorphTagger.SENTENCE_END_TAG or x == TrMorphTagger.SENTENCE_BEGIN_TAG:
                    continue
                elif len(x) == 1:
                    word_result.append(x)
                else:
                    word_result.append("+")
                    word_result.append(x)
            word_result = "".join(word_result)
            word_result = word_result.replace("+DB", "^DB")
            res.append(word_result)
        return res

    def train_model(self, model_name="model", early_stop=False, num_epoch=200):
        max_acc = 0.0
        epoch_loss = 0
        for epoch in range(num_epoch):
            random.shuffle(self.train)
            t1 = datetime.now()
            count = 0
            for i, sentence in enumerate(self.train, 1):
                loss_exp = self.get_loss(sentence)
                cur_loss = loss_exp.scalar_value()
                epoch_loss += cur_loss
                loss_exp.backward()
                self.trainer.update()

                # PRINT STATUS
                if i > 0 and i % 100 == 0:
                    t2 = datetime.now()
                    delta = t2 - t1
                    logger.info("loss = {}  /  {} instances finished in  {} seconds"
                                .format(epoch_loss / (i * 1.0), i, delta.seconds))
                count = i
            t2 = datetime.now()
            delta = t2 - t1
            logger.info("Epoch {} finished in {} minutes. loss = {}"
                        .format(epoch, delta.seconds / 60.0, epoch_loss / count * 1.0))

            epoch_loss = 0
            logger.info("Calculating Accuracy on dev set")
            acc, amb_acc = self.calculate_acc(self.dev)
            logger.info("Accuracy on dev set: {}  ambiguous accuracy on dev: ".format(acc, amb_acc))
            if acc > max_acc:
                max_acc = acc
                logger.info("Max accuracy increased = {}, saving model...".format(str(max_acc)))
                self.save_model(model_name)
            elif early_stop and max_acc - acc > 0.05:
                logger.info("Max accuracy did not increase, early stopping!")
                break

            logger.info("Calculating Accuracy on test sets")
            for q in range(len(self.test_paths)):
                logger.info("Calculating Accuracy on test set: {}".format(self.test_paths[q]))
                acc, amb_acc = self.calculate_acc(self.tests[q])
                logger.info(" accuracy: {}    ambiguous accuracy: {}".format(acc, amb_acc))

    def save_model(self, model_name):
        self.model.save("models/{}.model".format(model_name))
        with open("models/{}.char2id".format(model_name), "w") as f:
            pickle.dump(self.char2id, f)

    def load_model(self, model_name):
        self.model.load("models/" + model_name + ".model")
        with open("models/{}.char2id".format(model_name), "w") as f:
            self.char2id = pickle.load(self.char2id, f)
        self.id2char = {v: k for k, v in self.id2char.items()}

    def calculate_acc(self, sentences):
        corrects = 0
        non_ambigious_count = 0
        total = 0
        for sentence in sentences:
            predicted_labels = self.generate(sentence)
            gold_labels = ["".join(word.roots[0]) + "+" + "+".join(word.tags[0]) for word in sentence]
            gold_labels = [gold_label.replace("+DB", "^DB") for gold_label in gold_labels]
            for gold_label, predicted_label in zip(gold_labels, predicted_labels):
                if gold_label == predicted_label:
                    corrects += 1
            non_ambigious_count += [1 for w in sentence if len(w.roots) == 1].count(1)
            total += len(sentence)
        return (corrects * 1.0 / total), ((corrects - non_ambigious_count) * 1.0 / (total - non_ambigious_count))


if __name__ == "__main__":
    disambiguator = TrMorphTagger(train_from_scratch=True,
                                  train_data_path="data/data.train.txt",
                                  test_data_paths=["data/data.train.txt"],
                                  dev_data_path="data/data.train.txt",
                                  # test_data_paths=[
                                  #     "data/data.test.txt",
                                  #     "data/test.merge",
                                  #     "data/Morph.Dis.Test.Hand.Labeled-20K.txt"],
                                  model_file_name="encoder_decoder_morph_tagger")
