from transformers import DebertaForMaskedLM

class DebertaCustomModel(DebertaForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
