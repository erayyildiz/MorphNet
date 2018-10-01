import random
import re
import math
from collections import namedtuple
import dynet as dy
from datetime import datetime
import pickle
import logging.config

logging.config.fileConfig('resources/logging.ini')
logger = logging.getLogger(__file__)


class TrMorphTagger(object):

    # HYPER-PARAMETERS
    LSTM_NUM_OF_LAYERS = 1
    EMBEDDINGS_SIZE = 16
    OUTPUT_EMBEDDINGS_SIZE = 32
    STATE_SIZE = 64
    ATTENTION_SIZE = 64

    # STATIC VARIABLES
    EOS = "<s>"

    # COMPILED REGEX PATTERNS
    analysis_regex = re.compile(r"^([^\+]*)\+(.+)$", re.UNICODE)
    tag_seperator_regex = re.compile(r"[\+\^]", re.UNICODE)
    split_root_tags_regex = re.compile(r"^([^\+]+)\+(.+)$", re.IGNORECASE)

    # WORD STRUCT
    WordStruct = namedtuple("WordStruct", ["surface_word", "roots", "tags"])

    @classmethod
    def _create_vocab(cls, sentences):
        logger.info("Building Vocabulary...")
        char2id = dict()
        char2id[TrMorphTagger.EOS] = len(char2id)
        output_char2id = dict()
        output_char2id[TrMorphTagger.EOS] = len(output_char2id)
        for sentence in sentences:
            for word in sentence:
                for ch in word.surface_word:
                    if ch not in char2id:
                        char2id[ch] = len(char2id)
                for root in word.roots:
                    for ch in root:
                        if ch not in char2id:
                            char2id[ch] = len(char2id)
                        if ch not in output_char2id:
                            output_char2id[ch] = len(output_char2id)
                for tags in word.tags:
                    for tag in tags:
                        if tag not in output_char2id:
                            output_char2id[tag] = len(output_char2id)
        id2char = {v: k for k, v in output_char2id.items()}
        logger.info("Done.")
        return char2id, output_char2id, id2char

    @classmethod
    def _encode(cls, tokens, vocab):
        return [vocab[token] for token in tokens if token in vocab]

    def _embed(self, token, char_embedding_table):
        return [char_embedding_table[self.char2id[ch]] for ch in token]

    @staticmethod
    def lower(text):
        text = text.replace("Ü", "ü")
        text = text.replace("Ğ", "ğ")
        text = text.replace("İ", "i")
        text = text.replace("Ş", "ş")
        text = text.replace("Ç", "ç")
        text = text.replace("Ö", "ö")
        return text.lower()

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

            self.char2id, self.output_char2id, self.id2char = self._create_vocab(self.train)

            if not self.dev:
                train_size = int(math.floor(0.99 * len(self.train)))
                self.dev = self.train[train_size:]
                self.train = self.train[:train_size]

            self.model = dy.Model()
            self.trainer = dy.SimpleSGDTrainer(self.model, learning_rate=1.6)
            self.CHARS_LOOKUP = self.model.add_lookup_parameters((len(self.char2id),
                                                                  TrMorphTagger.EMBEDDINGS_SIZE))

            self.OUTPUT_LOOKUP = self.model.add_lookup_parameters((len(self.output_char2id),
                                                                   TrMorphTagger.OUTPUT_EMBEDDINGS_SIZE))

            self.enc_fwd_lstm = dy.LSTMBuilder(TrMorphTagger.LSTM_NUM_OF_LAYERS,
                                               TrMorphTagger.EMBEDDINGS_SIZE,
                                               TrMorphTagger.STATE_SIZE, self.model)
            self.enc_bwd_lstm = dy.LSTMBuilder(TrMorphTagger.LSTM_NUM_OF_LAYERS,
                                               TrMorphTagger.EMBEDDINGS_SIZE,
                                               TrMorphTagger.STATE_SIZE, self.model)

            self.dec_lstm = dy.LSTMBuilder(TrMorphTagger.LSTM_NUM_OF_LAYERS,
                                           TrMorphTagger.OUTPUT_EMBEDDINGS_SIZE,
                                           TrMorphTagger.STATE_SIZE * 2, self.model)

            self.decoder_w = self.model.add_parameters((len(self.output_char2id), TrMorphTagger.STATE_SIZE * 2))
            self.decoder_b = self.model.add_parameters((len(self.output_char2id)))

            self.train_model(model_name=model_file_name, num_epoch=10000)
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

    def load_data(self, file_path, max_sentence=10000000):
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

        states = s.add_inputs(input_vecs)
        rnn_outputs = [s.output() for s in states]
        return rnn_outputs

    def _get_word_representations(self, word_reps):
        fwd_rnn = self.enc_fwd_lstm.initial_state()
        fwd_vectors = self._run_rnn(fwd_rnn, word_reps)
        bwd_rnn = self.enc_bwd_lstm.initial_state()
        bwd_vectors = self._run_rnn(bwd_rnn, list(reversed(word_reps)))
        return dy.concatenate([fwd_vectors[0], bwd_vectors[-1]])

    def decode(self, word_rep, output):
        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        output = list(output) + [self.EOS]
        output = self._encode(output, self.output_char2id)

        last_output_embeddings = self.OUTPUT_LOOKUP[self.output_char2id[TrMorphTagger.EOS]]
        decoder_lstm = self.dec_lstm.initial_state().set_s([dy.tanh(word_rep),
                                                            dy.tanh(word_rep)])

        decoder_lstm = decoder_lstm.add_input(last_output_embeddings)
        losses = []

        for ch in output:
            decoder_lstm = decoder_lstm.add_input(last_output_embeddings)
            out_vector = w * decoder_lstm.output() + b
            probs = dy.softmax(out_vector)
            last_output_embeddings = self.OUTPUT_LOOKUP[ch]
            losses.append(-dy.log(dy.pick(probs, ch)))
        return losses

    def generate(self, word):
        dy.renew_cg()
        embedded = self._embed(word, self.CHARS_LOOKUP)
        word_rep = self._get_word_representations(embedded)

        w = dy.parameter(self.decoder_w)
        b = dy.parameter(self.decoder_b)

        last_output_embeddings = self.OUTPUT_LOOKUP[self.output_char2id[TrMorphTagger.EOS]]
        decoder_lstm = self.dec_lstm.initial_state().set_s([dy.tanh(word_rep),
                                                            dy.tanh(word_rep)])

        decoder_lstm = decoder_lstm.add_input(last_output_embeddings)

        out = []
        count_eos = 0
        for i in range(len(word) * 3):
            if count_eos == 1:
                break
            decoder_lstm = decoder_lstm.add_input(last_output_embeddings)
            out_vector = w * decoder_lstm.output() + b
            probs = dy.softmax(out_vector).vec_value()
            next_char = probs.index(max(probs))
            last_output_embeddings = self.OUTPUT_LOOKUP[next_char]
            if self.id2char[next_char] == TrMorphTagger.EOS:
                count_eos += 1
                continue

            out.append(self.id2char[next_char])
        return self._convert_to_morph_analyzer_form(out)

    def get_loss(self, word):
        embedded = self._embed(word.surface_word, self.CHARS_LOOKUP)
        encoded = self._get_word_representations(embedded)
        return self.decode(encoded, list(word.roots[0]) + word.tags[0])

    @staticmethod
    def _convert_to_morph_analyzer_form(sequence_arr):
        res = []
        for x in sequence_arr:
            if x == TrMorphTagger.EOS:
                continue
            elif len(x) == 1:
                res.append(x)
            else:
                res.append("+")
                res.append(x)
        res = "".join(res)
        res = res.replace("+DB", "^DB")
        return res

    def train_model(self, model_name="model", early_stop=False, num_epoch=200):
        max_acc = 0.0
        epoch_loss = 0
        for epoch in range(num_epoch):
            random.shuffle(self.train)
            t1 = datetime.now()
            count = 0
            for i, sentence in enumerate(self.train, 1):
                dy.renew_cg()
                losses = []
                for word in sentence:
                    losses += self.get_loss(word)
                loss = dy.esum(losses)
                cur_loss = loss.scalar_value()
                epoch_loss += cur_loss
                loss.backward()
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
        self.model.save("resources/models/{}.model".format(model_name))
        with open("resources/models/{}.char2id".format(model_name), "wb") as f:
            pickle.dump(self.char2id, f)

    def load_model(self, model_name):
        self.model.load("resources/models/" + model_name + ".model")
        with open("resources/models/{}.char2id".format(model_name), "rb") as f:
            self.char2id = pickle.load(self.char2id, f)
        self.id2char = {v: k for k, v in self.id2char.items()}

    def calculate_acc(self, sentences):
        corrects = 0
        non_ambigious_count = 0
        total = 0
        for sentence in sentences:
            for word in sentence:
                predicted_label = self.generate(word.surface_word)
                gold_label = word.roots[0] + "+" + "+".join(word.tags[0])
                gold_label = gold_label.replace("+DB", "^DB")
                logger.info(gold_label + " <==> " + predicted_label)
                if gold_label == predicted_label:
                    corrects += 1
                if len(word.roots) == 1:
                    non_ambigious_count += 1
                total += 1
        return (corrects * 1.0 / total), ((corrects - non_ambigious_count) * 1.0 / (total - non_ambigious_count))


if __name__ == "__main__":
    disambiguator = TrMorphTagger(train_from_scratch=True,
                                  train_data_path="data/data.train.txt",
                                  # test_data_paths=["data/data.train.txt"],
                                  dev_data_path="data/data.dev.txt",
                                  test_data_paths=[
                                      "data/data.test.txt",
                                      "data/test.merge",
                                      "data/Morph.Dis.Test.Hand.Labeled-20K.txt"],
                                  model_file_name="encoder_decoder_morph_tagger")
