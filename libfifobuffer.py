class ByteFIFO:
    """ byte FIFO buffer """
    def __init__(self):
        self._buf = bytearray()

    def put(self, data):
        self._buf.extend(data)

    def peek(self, size):
        return self._buf[:size]

    def take(self, size):
        data = self._buf[:size]
        # The fast delete syntax
        self._buf[:size] = b''
        return data

    def discard(self, size):
        self._buf[:size] = b''

    def getbuffer(self):
        # peek with no copy
        return self._buf

    def __len__(self):
        return len(self._buf)