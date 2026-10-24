"""Audit only successfully consumed SFT examples, including after resume."""
import hashlib
import struct


class SFTOrderAudit:
    def __init__(self, examples, state=None):
        self.examples = examples
        self.chain = '0' * 64
        self.seen = [0] * examples
        self.consumed = self.targets = self.epoch_targets = 0
        self.completed = []
        if state:
            if state['examples'] != examples:
                raise ValueError('SFT audit dataset size changed')
            self.__dict__.update(state)

    def add(self, indices, targets):
        if self.consumed % self.examples + len(indices) > self.examples:
            raise ValueError('Audited update crossed an epoch boundary')
        for index in indices:
            self.seen[index] += 1
        raw = struct.pack('<' + 'I' * len(indices), *indices)
        self.chain = hashlib.sha256(bytes.fromhex(self.chain) + raw).hexdigest()
        self.consumed += len(indices)
        self.targets += targets
        self.epoch_targets += targets
        if self.consumed % self.examples == 0:
            if any(n != 1 for n in self.seen):
                raise ValueError('SFT epoch did not cover every example exactly once')
            self.completed.append(dict(epoch=len(self.completed)+1, examples=self.examples,
                supervised_tokens=self.epoch_targets, order_sha256=self.chain,
                every_example_once=True))
            self.seen = [0] * self.examples
            self.epoch_targets = 0
            return True
        return False

    def state(self):
        return dict(self.__dict__)
