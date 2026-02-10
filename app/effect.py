class Effect:
    name = "BaseEffect"
    params = {}

    def apply(self, image, params):
        raise NotImplementedError("Each effect must implement apply()")
