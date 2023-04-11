from sentence_transformers import SentenceTransformer, util

class TopicSegmentation:
    def __init__(self, file, threshold=0.4, overlapping_topics=False):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.topic_threshold = threshold
        self.overlap_topics = overlapping_topics
        self.topics = []
        self.inputFile = file
        self.utterances = []
        self.cosine_scores = None
        self.embeddings = None
        if not self.overlap_topics:
            self.topics_inference()
        else:
            self.overlapping_topics_inference()


    def create_utterances(self):
        with open(self.inputFile, "rb") as f:
            self.utterances = f.read()

        self.utterances = self.utterances.decode("utf-8").splitlines()
        self.utterances = [x.split(":")[-1] for x in self.utterances]

        return self.utterances


    def create_sentence_embeddings(self):
        self.create_utterances()
        self.embeddings = self.model.encode(self.utterances, convert_to_tensor=True)

        return self.embeddings


    def compute_cosine_scores(self):
        self.create_sentence_embeddings()
        self.cosine_scores = util.cos_sim(self.embeddings, self.embeddings)
        self.cosine_scores = self.cosine_scores.tolist()

        return self.cosine_scores


    def topics_inference(self):
        self.compute_cosine_scores()
        topic_index = 0
        for i in range(len(self.cosine_scores)):
            if self.cosine_scores[i] != [-1]:
                self.topics.append([])
                for j in range(len(self.cosine_scores[i])):
                    if self.cosine_scores[i][j] >= self.topic_threshold and self.cosine_scores[j] != [-1]:
                        self.topics[topic_index].append(self.utterances[j])
                        if j != i:
                            self.cosine_scores[j] = [-1]
                topic_index += 1

        return self.topics

    def overlapping_topics_inference(self):
        self.compute_cosine_scores()
        topic_index = 0
        for i in range(len(self.cosine_scores)):
                self.topics.append([])
                for j in range(len(self.cosine_scores[i])):
                    if self.cosine_scores[i][j] >= self.topic_threshold:
                        self.topics[topic_index].append(self.utterances[j])
                topic_index += 1

        return self.topics


    def __repr__(self):
        counter = 0
        res = ""
        for x in self.topics:
            counter += 1
            res += ("Topic " + str(counter) + ":\n")
            for elem in x:
                res += (elem + "\n")
            res += "\n"
        return res



