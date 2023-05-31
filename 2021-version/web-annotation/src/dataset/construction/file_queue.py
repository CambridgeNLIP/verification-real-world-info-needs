import multiprocessing


class FileQueue():
    def __init__(self):

        manager = multiprocessing.Manager()
        self.q = manager.Queue(maxsize=2000)

    def enqueue(self, text):
        self.q.put(text)
